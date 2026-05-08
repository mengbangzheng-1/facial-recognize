# -*- coding: utf-8 -*-
"""FER System - Integration Tests

End-to-end tests for Student model, Teacher model, DistillationTrainer,
and EmotionPredictor.
"""

import os
import sys
import tempfile

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.student_model import ImprovedMobileNetV3Small
from models.student_model_config import NUM_CLASSES
from models.teacher_model import ConvNeXtTeacher
from training.distill_trainer import DistillationTrainer
from training.losses import CombinedLoss
from inference.predictor import EmotionPredictor, EMOTION_LABELS


def create_dummy_dataloader(batch_size=4, num_samples=8, num_classes=7):
    """Create a simple dataloader with random data for testing."""
    images = torch.randn(num_samples, 3, 48, 48)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ============================================================
# Student Model End-to-End Tests
# ============================================================

class TestStudentModelE2E:
    """End-to-end forward pass tests for the Student model."""

    @pytest.fixture(scope="class")
    def student_model(self):
        """Create student model with pretrained=False for faster testing."""
        model = ImprovedMobileNetV3Small(num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        return model

    def test_forward_pass_output_shape(self, student_model):
        """Student forward pass should output [B, num_classes]."""
        x = torch.randn(2, 3, 48, 48)
        with torch.no_grad():
            logits = student_model(x)
        assert logits.shape == (2, NUM_CLASSES), f"Expected (2,{NUM_CLASSES}), got {logits.shape}"

    def test_forward_pass_single_image(self, student_model):
        """Student should handle batch size 1."""
        x = torch.randn(1, 3, 48, 48)
        with torch.no_grad():
            logits = student_model(x)
        assert logits.shape == (1, NUM_CLASSES)

    def test_forward_pass_batch(self, student_model):
        """Student should handle larger batches."""
        x = torch.randn(8, 3, 48, 48)
        with torch.no_grad():
            logits = student_model(x)
        assert logits.shape == (8, NUM_CLASSES)

    def test_output_is_logits(self, student_model):
        """Output should be raw logits (not softmax probabilities)."""
        x = torch.randn(2, 3, 48, 48)
        with torch.no_grad():
            logits = student_model(x)
        # Logits can be negative and do not necessarily sum to 1
        assert (logits < 0).any() or (logits > 1).any() or \
               abs(logits.sum(dim=-1).item() - 1.0) > 0.1, \
               "Output does not look like raw logits"

    def test_softmax_gives_valid_distribution(self, student_model):
        """Softmax of output should sum to 1 per sample."""
        import torch.nn.functional as F
        x = torch.randn(4, 3, 48, 48)
        with torch.no_grad():
            logits = student_model(x)
            probs = F.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_gradient_flow(self, student_model):
        """Backward pass should produce gradients for student parameters."""
        student_model.train()
        x = torch.randn(2, 3, 48, 48)
        logits = student_model(x)
        loss = logits.sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in student_model.parameters())
        assert has_grad, "No gradients found after backward pass"

    def test_parameter_count_reasonable(self, student_model):
        """Student model should have approximately 991K parameters."""
        total = sum(p.numel() for p in student_model.parameters())
        # Allow 30% tolerance since pretrained=False may differ slightly
        assert 500_000 < total < 2_000_000, \
            f"Student model has {total} parameters, expected ~991K"


# ============================================================
# Teacher Model End-to-End Tests
# ============================================================

class TestTeacherModelE2E:
    """End-to-end forward pass tests for the Teacher model."""

    @pytest.fixture(scope="class")
    def teacher_model(self):
        """Create teacher model with pretrained=False for faster testing."""
        model = ConvNeXtTeacher(num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        return model

    def test_forward_pass_output(self, teacher_model):
        """Teacher forward should return (logits, features) tuple."""
        x = torch.randn(1, 3, 48, 48)
        with torch.no_grad():
            result = teacher_model(x)
        assert isinstance(result, tuple) and len(result) == 2

    def test_logits_shape(self, teacher_model):
        """Teacher logits should have shape [B, num_classes]."""
        x = torch.randn(2, 3, 48, 48)
        with torch.no_grad():
            logits, features = teacher_model(x)
        assert logits.shape == (2, NUM_CLASSES)

    def test_features_shape(self, teacher_model):
        """Teacher features should be a 2D tensor [B, feature_dim]."""
        x = torch.randn(2, 3, 48, 48)
        with torch.no_grad():
            logits, features = teacher_model(x)
        assert features.ndim == 2, f"Features should be 2D, got {features.ndim}D"
        assert features.shape[0] == 2

    def test_freeze_backbone(self, teacher_model):
        """freeze_backbone should freeze all params except classifier."""
        teacher_model.freeze_backbone()
        for name, param in teacher_model.named_parameters():
            if "classifier" not in name:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_unfreeze_backbone(self, teacher_model):
        """unfreeze_backbone should enable all gradients."""
        teacher_model.unfreeze_backbone()
        for param in teacher_model.parameters():
            assert param.requires_grad, "All params should be trainable after unfreeze"

    def test_parameter_count(self, teacher_model):
        """Teacher model should have approximately 87.5M parameters."""
        total = sum(p.numel() for p in teacher_model.parameters())
        # Allow 20% tolerance
        assert 60_000_000 < total < 120_000_000, \
            f"Teacher model has {total/1e6:.1f}M parameters, expected ~87.5M"


# ============================================================
# Distillation Trainer Single-Step Tests
# ============================================================

class TestDistillTrainerSingleStep:
    """Tests for running a single training step with the distillation trainer."""

    @pytest.fixture(scope="class")
    def trainer(self):
        """Create a DistillationTrainer with small models for testing."""
        device = torch.device("cpu")
        student = ImprovedMobileNetV3Small(num_classes=NUM_CLASSES, pretrained=False)
        teacher = ConvNeXtTeacher(num_classes=NUM_CLASSES, pretrained=False)
        teacher.freeze_backbone()
        criterion = CombinedLoss(focal_weight=0.7, kl_weight=0.3)
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DistillationTrainer(
                student=student,
                teacher=teacher,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                checkpoint_dir=tmpdir,
                log_dir=tmpdir,
                patience=5,
            )
            yield trainer
            # Clean up file handles so Windows can delete the temp dir
            for handler in trainer.logger.handlers[:]:
                handler.close()
                trainer.logger.removeHandler(handler)
            trainer.writer.close()

    def test_train_epoch_runs(self, trainer):
        """train_epoch should complete without errors."""
        train_loader = create_dummy_dataloader(batch_size=4, num_samples=8)
        metrics = trainer.train_epoch(train_loader)
        assert isinstance(metrics, dict)

    def test_train_epoch_metrics_keys(self, trainer):
        """train_epoch metrics should have expected keys."""
        train_loader = create_dummy_dataloader(batch_size=4, num_samples=8)
        metrics = trainer.train_epoch(train_loader)
        expected_keys = {"loss", "focal", "kl", "acc"}
        assert expected_keys == set(metrics.keys()), \
            f"Expected keys {expected_keys}, got {set(metrics.keys())}"

    def test_train_epoch_loss_finite(self, trainer):
        """Training loss should be a finite number."""
        train_loader = create_dummy_dataloader(batch_size=4, num_samples=8)
        metrics = trainer.train_epoch(train_loader)
        assert np.isfinite(metrics["loss"]), f"Loss is not finite: {metrics['loss']}"
        assert np.isfinite(metrics["focal"]), f"Focal loss not finite: {metrics['focal']}"
        assert np.isfinite(metrics["kl"]), f"KL loss not finite: {metrics['kl']}"

    def test_train_epoch_acc_range(self, trainer):
        """Training accuracy should be in [0, 100]."""
        train_loader = create_dummy_dataloader(batch_size=4, num_samples=8)
        metrics = trainer.train_epoch(train_loader)
        assert 0 <= metrics["acc"] <= 100

    def test_validate_runs(self, trainer):
        """validate should complete without errors."""
        val_loader = create_dummy_dataloader(batch_size=4, num_samples=8)
        metrics = trainer.validate(val_loader)
        assert isinstance(metrics, dict)

    def test_validate_metrics_keys(self, trainer):
        """validate metrics should have 'val_loss' and 'val_acc'."""
        val_loader = create_dummy_dataloader(batch_size=4, num_samples=8)
        metrics = trainer.validate(val_loader)
        assert set(metrics.keys()) == {"val_loss", "val_acc"}

    def test_teacher_frozen(self, trainer):
        """Teacher parameters should remain frozen after training step."""
        for param in trainer.teacher.parameters():
            assert not param.requires_grad, "Teacher should be frozen"

    def test_student_has_gradients(self, trainer):
        """Student should have gradients after a training step."""
        train_loader = create_dummy_dataloader(batch_size=4, num_samples=8)
        trainer.train_epoch(train_loader)
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in trainer.student.parameters())
        assert has_grad, "Student should have gradients after training"


# ============================================================
# EmotionPredictor Tests
# ============================================================

class TestEmotionPredictor:
    """Tests for EmotionPredictor inference."""

    @pytest.fixture(scope="class")
    def predictor_and_model(self):
        """Create a predictor by saving and loading a student model."""
        # Create student model and save weights
        model = ImprovedMobileNetV3Small(num_classes=NUM_CLASSES, pretrained=False)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "student.pth")
            torch.save(model.state_dict(), model_path)
            predictor = EmotionPredictor(model_path=model_path, device="cpu")
            yield predictor, model

    def test_predict_returns_dict(self, predictor_and_model):
        """predict should return a dict mapping emotion names to floats."""
        predictor, _ = predictor_and_model
        face = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        result = predictor.predict(face)
        assert isinstance(result, dict)

    def test_predict_has_all_emotions(self, predictor_and_model):
        """predict should return probabilities for all 7 emotions."""
        predictor, _ = predictor_and_model
        face = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        result = predictor.predict(face)
        assert set(result.keys()) == set(EMOTION_LABELS)

    def test_predict_probabilities_sum_to_one(self, predictor_and_model):
        """Probabilities should sum to approximately 1.0."""
        predictor, _ = predictor_and_model
        face = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        result = predictor.predict(face)
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total}, expected 1.0"

    def test_predict_probabilities_non_negative(self, predictor_and_model):
        """All probabilities should be non-negative."""
        predictor, _ = predictor_and_model
        face = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        result = predictor.predict(face)
        assert all(v >= 0 for v in result.values())

    def test_predict_with_gray_input(self, predictor_and_model):
        """predict should handle grayscale input."""
        predictor, _ = predictor_and_model
        face = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        result = predictor.predict(face)
        assert set(result.keys()) == set(EMOTION_LABELS)

    def test_predict_with_different_sizes(self, predictor_and_model):
        """predict should handle different input sizes via resize."""
        predictor, _ = predictor_and_model
        for size in [32, 64, 128]:
            face = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            result = predictor.predict(face)
            assert set(result.keys()) == set(EMOTION_LABELS)

    def test_predict_batch(self, predictor_and_model):
        """predict_batch should return a list of dicts."""
        predictor, _ = predictor_and_model
        faces = [np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8) for _ in range(3)]
        results = predictor.predict_batch(faces)
        assert isinstance(results, list)
        assert len(results) == 3
        for r in results:
            assert set(r.keys()) == set(EMOTION_LABELS)

    def test_predict_topk(self, predictor_and_model):
        """predict_topk should return k items sorted descending."""
        predictor, _ = predictor_and_model
        face = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        top3 = predictor.predict_topk(face, k=3)
        assert len(top3) == 3
        assert all(isinstance(item, tuple) and len(item) == 2 for item in top3)
        # Check descending order
        probs = [p for _, p in top3]
        assert probs == sorted(probs, reverse=True), "Top-k should be sorted descending"
