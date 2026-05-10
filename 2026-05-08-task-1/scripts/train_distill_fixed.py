"""
FER2013 知识蒸馏训练脚本
学生模型: ImprovedMobileNetV3Small (SE + CBAM + ASPP) + ConvNeXt-Base 教师蒸馏

修复说明:
  MobileNetV3 Small backbone 各层输出通道 (参考 torchvision.models.mobilenetv3.py):
    backbone[0]   = 16  (Stem conv: 3 -> 16)
    backbone[1]   = 16  (Layer 1 "C1": stride=2, SE=True, RE)
    backbone[2]   = 24  (Layer 2 "C2": stride=2, SE=False, RE)
    backbone[3]   = 24  (Layer 3: stride=1, SE=False, RE)
    backbone[4]   = 40  (Layer 4 "C3": stride=2, SE=True, HS)
    backbone[5]   = 40  (Layer 5: stride=1, SE=True, HS)
    backbone[6]   = 40  (Layer 6: stride=1, SE=True, HS)  ← 注意是 40 不是 48!
    backbone[7]   = 48  (Layer 7: stride=1, SE=True, HS)
    backbone[8]   = 48  (Layer 8: stride=1, SE=True, HS)
    backbone[9]   = 96  (Layer 9 "C4": stride=2, SE=True, HS)
    backbone[10]  = 96  (Layer 10: stride=1, SE=True, HS)
    backbone[11]  = 96  (Layer 11: stride=1, SE=True, HS)
    backbone[12]  = 576 (Final conv: 96 -> 576, Hardswish)

  原代码错误: SEBlock(48) / CBAM(576) 用于 backbone[4:6] 输出 (40ch)
  正确:        SEBlock(40) / CBAM(48) 用于 backbone[6:9] 输出
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import pandas as pd
import numpy as np

# ===================== 配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "data/fer2013/fer2013.csv"
TEACHER_CKPT = "model_checkpoints/teacher/best_teacher.pth"
BATCH_SIZE = 256
EPOCHS = 80
LR = 3e-4
NUM_CLASSES = 7
IMAGE_SIZE = 224

# ===================== FER2013 数据集 =====================
class FER2013Dataset(Dataset):
    def __init__(self, csv_path, split="Training", transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["Usage"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pixels = self.df.iloc[idx]["pixels"]
        img = np.fromstring(pixels, sep=" ").reshape(48, 48).astype(np.uint8)
        img = np.stack([img] * 3, axis=-1)  # 灰度 -> RGB 3通道
        img = transforms.ToPILImage()(img)
        label = int(self.df.iloc[idx]["emotion"])
        if self.transform:
            img = self.transform(img)
        return img, label

# ===================== 数据增强 =====================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = FER2013Dataset(CSV_PATH, "Training", train_transform)
val_dataset   = FER2013Dataset(CSV_PATH, "PublicTest", val_transform)

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ===================== 加载教师模型 (ConvNeXt-Base) =====================
teacher = models.convnext_base(weights=None)
teacher.classifier[-1] = nn.Linear(1024, NUM_CLASSES)
teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location=DEVICE), strict=False)
teacher = teacher.to(DEVICE).eval()
for p in teacher.parameters():
    p.requires_grad = False

print(f"✅ 教师模型加载完成: ConvNeXt-Base | 设备: {DEVICE}")

# ===================== 注意力模块: SE + CBAM + ASPP =====================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 模块"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CBAM(nn.Module):
    """Convolutional Block Attention Module: Channel + Spatial Attention"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        x = x * self.ca(x)
        # Spatial attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa_map   = torch.cat([max_pool, avg_pool], dim=1)
        x = x * self.sa(sa_map)
        return x


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling: 多尺度空洞卷积"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.c2 = nn.Conv2d(in_channels, out_channels, 3,
                            padding=6, dilation=6, bias=False)
        self.c3 = nn.Conv2d(in_channels, out_channels, 3,
                            padding=12, dilation=12, bias=False)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.bn  = nn.BatchNorm2d(out_channels * 4)
        self.out = nn.Conv2d(out_channels * 4, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        y1 = self.c1(x)
        y2 = self.c2(x)
        y3 = self.c3(x)
        y4 = self.pool(x).repeat(1, 1, h, w)
        concat = torch.cat([y1, y2, y3, y4], dim=1)
        return self.out(self.bn(concat))


# ===================== ✅ 修正后的学生模型 =====================
# 通道完全对齐 MobileNetV3 Small 源码
class StudentModel(nn.Module):
    """
    Improved MobileNetV3-Small 学生模型
    - SE Block: 通道注意力，强化特征选择
    - CBAM:     通道 + 空间双重注意力
    - ASPP:     多尺度感知，捕获不同感受野

    Backbone 切片与通道映射 (参考 torchvision.models.mobilenetv3.py):
      backbone[0]  -> 16ch  (Stem)
      backbone[1]   -> 16ch  (C1: stride=2, SE)
      backbone[2]  -> 24ch  (C2: stride=2)
      backbone[3]  -> 24ch  (stride=1)
      backbone[4]  -> 40ch  (C3: stride=2, SE, HS)
      backbone[5]  -> 40ch  (stride=1, SE, HS)
      backbone[6]  -> 40ch  (stride=1, SE, HS)  ← 40ch，非 48ch！
      backbone[7]  -> 48ch  (stride=1, SE, HS)
      backbone[8]  -> 48ch  (stride=1, SE, HS)
      backbone[9]  -> 96ch  (C4: stride=2, SE, HS)
      backbone[10] -> 96ch  (stride=1, SE, HS)
      backbone[11] -> 96ch  (stride=1, SE, HS)
      backbone[12] -> 576ch (Final conv: 96->576, HS)
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()
        mob = models.mobilenet_v3_small(weights="DEFAULT")
        self.backbone = mob.features

        # ---- 修正后的注意力模块，通道数 100% 对齐 ----
        self.se1   = SEBlock(16)   # backbone[0:2]  -> 16ch
        self.cbam1 = CBAM(16)     # backbone[0:2]  -> 16ch

        self.se2   = SEBlock(24)   # backbone[2:4]  -> 24ch
        self.cbam2 = CBAM(24)     # backbone[2:4]  -> 24ch

        self.se3   = SEBlock(40)   # backbone[4:7]  -> 40ch  ← 修正: 40 而非 48
        self.cbam3 = CBAM(48)     # backbone[7:9]  -> 48ch  ← 修正: 48 而非 576

        self.aspp = ASPP(576, 128)  # backbone[9:]+conv -> 576ch

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: 16ch -> SE + CBAM
        x = self.backbone[0:2](x)
        x = self.se1(x)
        x = self.cbam1(x)

        # Stage 2: 24ch -> SE + CBAM
        x = self.backbone[2:4](x)
        x = self.se2(x)
        x = self.cbam2(x)

        # Stage 3: 40ch -> SE
        x = self.backbone[4:7](x)    # backbone[4:6] = 40ch, backbone[6] = 40ch
        x = self.se3(x)

        # Stage 4: 48ch -> CBAM -> ASPP
        x = self.backbone[7:9](x)     # backbone[7] = 48ch, backbone[8] = 48ch
        x = self.cbam3(x)

        # Stage 5: 576ch -> ASPP -> Classifier
        x = self.backbone[9:](x)     # backbone[9:12] = 96ch, backbone[12] = 576ch
        x = self.aspp(x)

        return self.classifier(x)


# ===================== 蒸馏损失函数 =====================
class DistillLoss(nn.Module):
    """
    知识蒸馏损失: Hard Label CrossEntropy + Soft Label KL Divergence
    - Temperature T 控制软标签的平滑程度
    - kl_weight 控制蒸馏信号强度
    """
    def __init__(self, temperature: float = 4.0,
                 focal_weight: float = 0.5, kl_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.temp = temperature
        self.focal_weight = focal_weight
        self.kl_weight    = kl_weight

    def forward(self, s_logit: torch.Tensor,
                t_logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        hard_loss = self.ce(s_logit, label)
        soft_loss = self.kl(
            torch.log_softmax(s_logit / self.temp, dim=1),
            torch.softmax(t_logit / self.temp, dim=1)
        ) * (self.temp ** 2)
        return self.focal_weight * hard_loss + self.kl_weight * soft_loss


# ===================== 训练循环 =====================
def train_epoch(model, teacher, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(train_loader, desc="Training")
    for img, label in pbar:
        img, label = img.to(device), label.to(device)
        with torch.no_grad():
            t_out = teacher(img)
        s_out = model(img)
        loss  = criterion(s_out, t_out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * img.size(0)
        correct     += (s_out.argmax(1) == label).sum().item()
        total       += img.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc":  f"{100 * correct / total:.2f}%"
        })

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    for img, label in val_loader:
        img, label = img.to(device), label.to(device)
        correct += (model(img).argmax(1) == label).sum().item()
        total   += img.size(0)
    return 100.0 * correct / total


# ===================== 入口 =====================
def main():
    model    = StudentModel(NUM_CLASSES).to(DEVICE)
    criterion = DistillLoss(temperature=4.0, focal_weight=0.5, kl_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0

    print(f"\n{'='*60}")
    print("✅ 学生模型启动: MobileNetV3-Small + SE + CBAM + ASPP")
    print(f"   设备: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"   温度: 4.0 | KL权重: 0.5 | CE权重: 0.5")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(
            model, teacher, train_loader, criterion, optimizer, DEVICE)
        val_acc = validate(model, val_loader, DEVICE)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_student.pth")
            marker = " ✅ 最佳模型保存"
        else:
            marker = ""

        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train: {train_acc:.2f}% | "
            f"Val: {val_acc:.2f}% | "
            f"Best: {best_acc:.2f}%{marker}"
        )

    print(f"\n训练完成！最佳验证准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
