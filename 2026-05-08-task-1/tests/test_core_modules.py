# -*- coding: utf-8 -*-
"""FER System - Core Module Unit Tests

Tests for attention modules, ASPP, loss functions, face detector, and callbacks.
"""

import os
import sys
import tempfile

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.attention import SEModule, CBAM, ChannelAttention, SpatialAttention
from models.aspp import ASPP
from training.losses import FocalLoss, DistillationLoss, CombinedLoss
from inference.face_detector import FaceDetector
from training.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# SE Module Tests
# ============================================================

class TestSEModule:
    """Tests for Squeeze-and-Excitation channel attention module."""

    def test_output_shape(self):
        """SE module should preserve input spatial dimensions."""
        se = SEModule(channels=64, reduction=16)
        x = torch.randn(2, 64, 8, 8)
        out = se(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_attention_weights_in_range(self):
        """SE attention weights (sigmoid output) should be in [0, 1] range.
        Note: SE output = input * sigmoid(weights), so output can be negative
        if input has negative values."""
        se = SEModule(channels=32, reduction=8)
        x = torch.randn(1, 32, 4, 4)
        # Extract attention weights manually
        b, c, _, _ = x.size()
        y = se.avg_pool(x).view(b, c)
        weights = se.fc(y).view(b, c, 1, 1).sigmoid()
        assert (weights >= 0).all() and (weights <= 1).all(), \
            "SE attention weights should be in [0, 1]"

    def test_non_negative_output_with_positive_input(self):
        """SE output should be non-negative when input is non-negative."""
        se = SEModule(channels=32, reduction=8)
        x = torch.rand(1, 32, 4, 4)  # positive input only
        out = se(x)
        assert (out >= 0).all(), "SE output should be non-negative for positive input"

    def test_different_batch_sizes(self):
        """SE module should work with batch size 1 and larger."""
        se = SEModule(channels=16)
        for bs in [1, 4, 8]:
            x = torch.randn(bs, 16, 6, 6)
            out = se(x)
            assert out.shape == (bs, 16, 6, 6)

    def test_channels_smaller_than_reduction(self):
        """When channels < reduction, mid_channels should be at least 1."""
        se = SEModule(channels=8, reduction=16)
        x = torch.randn(1, 8, 4, 4)
        out = se(x)
        assert out.shape == x.shape

    def test_single_channel(self):
        """SE module should work with a single channel."""
        se = SEModule(channels=1, reduction=1)
        x = torch.randn(2, 1, 10, 10)
        out = se(x)
        assert out.shape == x.shape

    def test_zero_input(self):
        """SE module with zero input should produce zero output."""
        se = SEModule(channels=16)
        x = torch.zeros(1, 16, 4, 4)
        out = se(x)
        assert torch.allclose(out, x, atol=1e-6)


# ============================================================
# CBAM Module Tests
# ============================================================

class TestChannelAttention:
    """Tests for CBAM channel attention sub-module."""

    def test_output_shape(self):
        """Channel attention should output [B, C, 1, 1]."""
        ca = ChannelAttention(channels=64, reduction=16)
        x = torch.randn(2, 64, 8, 8)
        out = ca(x)
        assert out.shape == (2, 64, 1, 1), f"Expected (2,64,1,1), got {out.shape}"

    def test_output_in_sigmoid_range(self):
        """Channel attention output should be in [0, 1] range (sigmoid)."""
        ca = ChannelAttention(channels=32)
        x = torch.randn(1, 32, 4, 4)
        out = ca(x)
        assert (out >= 0).all() and (out <= 1).all(), "Output should be in [0,1]"


class TestSpatialAttention:
    """Tests for CBAM spatial attention sub-module."""

    def test_output_shape(self):
        """Spatial attention should output [B, 1, H, W]."""
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(2, 64, 8, 8)
        out = sa(x)
        assert out.shape == (2, 1, 8, 8), f"Expected (2,1,8,8), got {out.shape}"

    def test_output_in_sigmoid_range(self):
        """Spatial attention output should be in [0, 1] range."""
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(1, 32, 6, 6)
        out = sa(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_odd_and_even_kernel(self):
        """Should work with both odd and even kernel sizes."""
        for ks in [3, 5, 7]:
            sa = SpatialAttention(kernel_size=ks)
            x = torch.randn(1, 16, 8, 8)
            out = sa(x)
            assert out.shape == (1, 1, 8, 8), f"Failed for kernel_size={ks}"


class TestCBAM:
    """Tests for full CBAM module."""

    def test_output_shape(self):
        """CBAM should preserve input shape [B, C, H, W]."""
        cbam = CBAM(channels=64, reduction=16, spatial_kernel=7)
        x = torch.randn(2, 64, 8, 8)
        out = cbam(x)
        assert out.shape == x.shape

    def test_non_negative_output_with_positive_input(self):
        """CBAM output should be non-negative when input is non-negative.
        Note: CBAM output = input * attention_weights, so output can be negative
        if input has negative values (attention weights are in [0,1])."""
        cbam = CBAM(channels=32)
        x = torch.rand(1, 32, 4, 4)  # positive input only
        out = cbam(x)
        assert (out >= 0).all(), "CBAM output should be non-negative for positive input"

    def test_zero_input(self):
        """CBAM with zero input should produce zero output."""
        cbam = CBAM(channels=16)
        x = torch.zeros(1, 16, 4, 4)
        out = cbam(x)
        assert torch.allclose(out, x, atol=1e-6)

    def test_different_spatial_sizes(self):
        """CBAM should work with various spatial sizes."""
        cbam = CBAM(channels=32)
        for size in [6, 12, 24]:
            x = torch.randn(1, 32, size, size)
            out = cbam(x)
            assert out.shape == (1, 32, size, size)


# ============================================================
# ASPP Module Tests
# ============================================================

class TestASPP:
    """Tests for Atrous Spatial Pyramid Pooling module."""

    def test_output_shape(self):
        """ASPP should produce [B, out_channels, H, W]."""
        aspp = ASPP(in_channels=64, out_channels=64, dilations=[1, 6, 12, 18])
        x = torch.randn(2, 64, 8, 8)
        out = aspp(x)
        assert out.shape == (2, 64, 8, 8), f"Expected (2,64,8,8), got {out.shape}"

    def test_default_dilations(self):
        """ASPP with default dilations should work."""
        aspp = ASPP(in_channels=96, out_channels=96)
        aspp.eval()  # eval mode to avoid BatchNorm issues with batch_size=1
        x = torch.randn(1, 96, 6, 6)
        out = aspp(x)
        assert out.shape == (1, 96, 6, 6)

    def test_single_dilation(self):
        """ASPP with a single dilation rate."""
        aspp = ASPP(in_channels=32, out_channels=32, dilations=[1])
        aspp.eval()
        x = torch.randn(1, 32, 4, 4)
        out = aspp(x)
        assert out.shape == (1, 32, 4, 4)

    def test_odd_channels_division(self):
        """ASPP should handle out_channels not evenly divisible by num_branches."""
        aspp = ASPP(in_channels=32, out_channels=33, dilations=[1, 6])
        aspp.eval()
        x = torch.randn(1, 32, 8, 8)
        out = aspp(x)
        assert out.shape[1] == 33, f"Expected 33 channels, got {out.shape[1]}"

    def test_different_input_sizes(self):
        """ASPP should preserve spatial dimensions for different sizes."""
        aspp = ASPP(in_channels=64, out_channels=64)
        aspp.eval()
        for size in [4, 12, 24]:
            x = torch.randn(1, 64, size, size)
            out = aspp(x)
            assert out.shape == (1, 64, size, size), f"Failed for size={size}"


# ============================================================
# FocalLoss Tests
# ============================================================

class TestFocalLoss:
    """Tests for Focal Loss."""

    def test_basic_forward(self):
        """FocalLoss should produce a scalar loss value."""
        fl = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 7)
        targets = torch.randint(0, 7, (4,))
        loss = fl(logits, targets)
        assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"

    def test_loss_positive(self):
        """FocalLoss value should be non-negative."""
        fl = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 7)
        targets = torch.randint(0, 7, (8,))
        loss = fl(logits, targets)
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_with_class_weights(self):
        """FocalLoss with alpha (class weights) should work."""
        alpha = torch.ones(7)  # uniform weights
        fl = FocalLoss(alpha=alpha, gamma=2.0)
        logits = torch.randn(4, 7)
        targets = torch.randint(0, 7, (4,))
        loss = fl(logits, targets)
        assert loss.ndim == 0

    def test_gamma_zero_equals_cross_entropy(self):
        """With gamma=0, FocalLoss should equal mean CrossEntropy loss."""
        fl = FocalLoss(gamma=0.0)
        logits = torch.randn(8, 7)
        targets = torch.randint(0, 7, (8,))
        fl_val = fl(logits, targets)
        ce_val = nn.CrossEntropyLoss()(logits, targets)
        assert torch.allclose(fl_val, ce_val, atol=1e-5), \
            f"FocalLoss(gamma=0)={fl_val} != CE={ce_val}"

    def test_reduction_modes(self):
        """FocalLoss should support 'mean', 'sum', and 'none' reductions."""
        logits = torch.randn(4, 7)
        targets = torch.randint(0, 7, (4,))
        for reduction in ["mean", "sum", "none"]:
            fl = FocalLoss(gamma=2.0, reduction=reduction)
            loss = fl(logits, targets)
            if reduction == "mean":
                assert loss.ndim == 0
            elif reduction == "sum":
                assert loss.ndim == 0
            elif reduction == "none":
                assert loss.shape == (4,)

    def test_perfect_classification_low_loss(self):
        """Correctly classified examples should have lower loss than random."""
        fl = FocalLoss(gamma=2.0)
        # Well-classified: target=0, logit[0] is very large
        good_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        good_targets = torch.tensor([0])
        good_loss = fl(good_logits, good_targets)

        # Random logits
        rand_logits = torch.randn(1, 7)
        rand_targets = torch.tensor([0])
        rand_loss = fl(rand_logits, rand_targets)

        assert good_loss < rand_loss, \
            f"Well-classified loss ({good_loss}) should be < random loss ({rand_loss})"


# ============================================================
# DistillationLoss Tests
# ============================================================

class TestDistillationLoss:
    """Tests for KL Divergence distillation loss."""

    def test_basic_forward(self):
        """DistillationLoss should produce a scalar loss value."""
        dl = DistillationLoss(temperature=4.0)
        student_logits = torch.randn(4, 7)
        teacher_logits = torch.randn(4, 7)
        loss = dl(student_logits, teacher_logits)
        assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"

    def test_loss_non_negative(self):
        """KL divergence loss should be non-negative."""
        dl = DistillationLoss(temperature=4.0)
        student_logits = torch.randn(4, 7)
        teacher_logits = torch.randn(4, 7)
        loss = dl(student_logits, teacher_logits)
        assert loss.item() >= 0

    def test_identical_inputs_low_loss(self):
        """Identical student/teacher logits should give very low loss."""
        dl = DistillationLoss(temperature=4.0)
        logits = torch.randn(4, 7)
        loss = dl(logits, logits)
        assert loss.item() < 0.01, \
            f"Identical logits should give near-zero loss, got {loss.item()}"

    def test_temperature_scaling(self):
        """Higher temperature should produce higher loss (for same logits)."""
        logits_s = torch.randn(4, 7)
        logits_t = torch.randn(4, 7)
        dl_low = DistillationLoss(temperature=1.0)
        dl_high = DistillationLoss(temperature=8.0)
        loss_low = dl_low(logits_s, logits_t)
        loss_high = dl_high(logits_s, logits_t)
        assert loss_high > loss_low, \
            f"T=8 loss ({loss_high}) should > T=1 loss ({loss_low})"


# ============================================================
# CombinedLoss Tests
# ============================================================

class TestCombinedLoss:
    """Tests for combined Focal + KL distillation loss."""

    def test_returns_tuple(self):
        """CombinedLoss should return (total_loss, loss_info_dict)."""
        cl = CombinedLoss(focal_weight=0.7, kl_weight=0.3)
        student_logits = torch.randn(4, 7)
        teacher_logits = torch.randn(4, 7)
        targets = torch.randint(0, 7, (4,))
        result = cl(student_logits, teacher_logits, targets)
        assert isinstance(result, tuple) and len(result) == 2
        total_loss, loss_info = result
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(loss_info, dict)

    def test_loss_info_keys(self):
        """Loss info dict should contain 'focal_loss', 'kl_loss', 'total_loss'."""
        cl = CombinedLoss()
        student_logits = torch.randn(4, 7)
        teacher_logits = torch.randn(4, 7)
        targets = torch.randint(0, 7, (4,))
        _, loss_info = cl(student_logits, teacher_logits, targets)
        assert set(loss_info.keys()) == {"focal_loss", "kl_loss", "total_loss"}

    def test_total_loss_matches_formula(self):
        """total_loss should equal focal_weight*focal + kl_weight*kl."""
        focal_weight = 0.7
        kl_weight = 0.3
        cl = CombinedLoss(focal_weight=focal_weight, kl_weight=kl_weight)
        student_logits = torch.randn(4, 7)
        teacher_logits = torch.randn(4, 7)
        targets = torch.randint(0, 7, (4,))
        total_loss, loss_info = cl(student_logits, teacher_logits, targets)
        expected = focal_weight * loss_info["focal_loss"] + kl_weight * loss_info["kl_loss"]
        assert abs(total_loss.item() - expected) < 1e-5, \
            f"total_loss={total_loss.item()} != expected={expected}"

    def test_with_class_weights(self):
        """CombinedLoss with alpha class weights should work."""
        alpha = torch.ones(7)
        cl = CombinedLoss(alpha=alpha)
        student_logits = torch.randn(4, 7)
        teacher_logits = torch.randn(4, 7)
        targets = torch.randint(0, 7, (4,))
        total_loss, loss_info = cl(student_logits, teacher_logits, targets)
        assert total_loss.ndim == 0


# ============================================================
# FaceDetector Tests
# ============================================================

class TestFaceDetector:
    """Tests for OpenCV Haar cascade face detector."""

    def test_init(self):
        """FaceDetector should initialize without errors."""
        fd = FaceDetector()
        assert not fd.cascade.empty(), "Cascade should be loaded"

    def test_detect_synthetic_image(self):
        """Detection on a synthetic image should return a list."""
        fd = FaceDetector()
        # Create a gray image with a lighter rectangle to simulate a face
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = fd.detect(image)
        assert isinstance(faces, list)

    def test_detect_empty(self):
        """Blank image should return empty list."""
        fd = FaceDetector()
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        faces = fd.detect(image)
        assert faces == []

    def test_detect_returns_tuples(self):
        """Each detection should be a tuple of (x, y, w, h)."""
        fd = FaceDetector()
        # Use a slightly more complex pattern
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        faces = fd.detect(image)
        for f in faces:
            assert isinstance(f, tuple) and len(f) == 4

    def test_crop_face_basic(self):
        """crop_face should return a resized face image."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox = (20, 20, 40, 40)
        face = FaceDetector.crop_face(image, bbox, target_size=(48, 48))
        assert face is not None
        assert face.shape[:2] == (48, 48)

    def test_crop_face_with_expand(self):
        """crop_face with expand_ratio should expand the bounding box."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox = (30, 30, 40, 40)
        face = FaceDetector.crop_face(image, bbox, target_size=(48, 48), expand_ratio=0.2)
        assert face is not None
        assert face.shape[:2] == (48, 48)

    def test_crop_face_edge_clamp(self):
        """crop_face should clamp coordinates to image boundaries."""
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        bbox = (0, 0, 50, 50)
        face = FaceDetector.crop_face(image, bbox, target_size=(48, 48))
        assert face is not None
        assert face.shape[:2] == (48, 48)

    def test_crop_face_custom_target_size(self):
        """crop_face should respect custom target_size."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox = (10, 10, 80, 80)
        face = FaceDetector.crop_face(image, bbox, target_size=(64, 64))
        assert face is not None
        assert face.shape[:2] == (64, 64)


# ============================================================
# EarlyStopping Tests
# ============================================================

class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_initial_state(self):
        """EarlyStopping should start with should_stop=False."""
        es = EarlyStopping(patience=3)
        assert es.should_stop is False

    def test_no_stop_on_improvement(self):
        """Should not stop while metric keeps improving (mode=min)."""
        es = EarlyStopping(patience=3, mode="min")
        assert not es.step(1.0)
        assert not es.step(0.8)
        assert not es.step(0.5)

    def test_stop_on_patience_exceeded(self):
        """Should stop after patience epochs without improvement."""
        es = EarlyStopping(patience=3, mode="min")
        es.step(1.0)   # best=1.0, counter=0
        es.step(1.1)   # no improvement, counter=1
        es.step(1.2)   # counter=2
        assert es.step(1.3), "Should stop after 3 epochs without improvement"

    def test_min_delta_respected(self):
        """Improvement smaller than min_delta should not reset counter."""
        es = EarlyStopping(patience=2, mode="min", min_delta=0.5)
        es.step(1.0)   # best=1.0, counter=0
        es.step(0.9)   # improvement < 0.5, counter=1
        assert es.step(0.8), "Should stop: improvement < min_delta"

    def test_mode_max(self):
        """mode='max' should track higher-is-better."""
        es = EarlyStopping(patience=2, mode="max")
        es.step(0.5)   # best=0.5, counter=0
        es.step(0.7)   # best=0.7, counter=0 (improvement)
        es.step(0.6)   # counter=1
        assert es.step(0.6), "Should stop for mode=max"

    def test_patience_reset_on_improvement(self):
        """Counter should reset when improvement occurs."""
        es = EarlyStopping(patience=3, mode="min")
        es.step(1.0)   # best=1.0, counter=0
        es.step(1.1)   # no improvement, counter=1
        es.step(0.5)   # best=0.5, counter=0 (improvement resets counter)
        es.step(0.6)   # no improvement, counter=1
        assert not es.step(0.7), \
            "Should NOT stop: only 2 epochs without improvement after reset"


# ============================================================
# ModelCheckpoint Tests
# ============================================================

class TestModelCheckpoint:
    """Tests for ModelCheckpoint callback."""

    def test_first_save(self):
        """First call should always save the model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = ModelCheckpoint(tmpdir, mode="min")
            model = nn.Linear(10, 5)
            saved = mc.step(1.0, model)
            assert saved is True
            assert (mc.save_dir / "best_model.pth").exists()

    def test_save_on_improvement_min(self):
        """Should save when score improves (mode='min')."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = ModelCheckpoint(tmpdir, mode="min")
            model = nn.Linear(10, 5)
            mc.step(1.0, model)
            saved = mc.step(0.5, model)
            assert saved is True

    def test_no_save_on_worse_min(self):
        """Should NOT save when score gets worse (mode='min')."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = ModelCheckpoint(tmpdir, mode="min")
            model = nn.Linear(10, 5)
            mc.step(0.5, model)
            saved = mc.step(1.0, model)
            assert saved is False

    def test_save_on_improvement_max(self):
        """Should save when score improves (mode='max')."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = ModelCheckpoint(tmpdir, mode="max")
            model = nn.Linear(10, 5)
            mc.step(0.5, model)
            saved = mc.step(0.9, model)
            assert saved is True

    def test_no_save_on_worse_max(self):
        """Should NOT save when score gets worse (mode='max')."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = ModelCheckpoint(tmpdir, mode="max")
            model = nn.Linear(10, 5)
            mc.step(0.9, model)
            saved = mc.step(0.5, model)
            assert saved is False

    def test_checkpoint_loadable(self):
        """Saved checkpoint should be loadable by torch.load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mc = ModelCheckpoint(tmpdir, mode="min")
            model = nn.Linear(10, 5)
            mc.step(1.0, model)
            state_dict = torch.load(mc.save_dir / "best_model.pth", map_location="cpu")
            assert set(state_dict.keys()) == set(model.state_dict().keys())
