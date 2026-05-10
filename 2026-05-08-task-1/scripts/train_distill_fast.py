"""
FER2013 知识蒸馏训练脚本 (预计算 Logits 快速版)
训练时直接读取离线教师 logits，不再每 batch 跑 ConvNeXt-Base
速度提升: ~10min/epoch -> ~1-2min/epoch

前提: 先运行 scripts/precompute_teacher_logits.py 生成:
    data/fer2013/teacher_logits_train.pt
    data/fer2013/teacher_logits_val.pt
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import csv
import numpy as np

# ===================== 配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH    = "data/fer2013/fer2013.csv"
TRAIN_LOGITS_PATH = "data/fer2013/teacher_logits_train.pt"
VAL_LOGITS_PATH   = "data/fer2013/teacher_logits_val.pt"
BATCH_SIZE  = 256
EPOCHS      = 80
LR          = 3e-4
NUM_CLASSES = 7
IMAGE_SIZE  = 224

# ===================== 数据集 (带离线 Teacher Logits) =====================
class FER2013WithLogits(Dataset):
    """
    训练集: 返回 (img, label, teacher_logit)
    teacher_logit 是预计算好的教师输出，形状 (7,)
    注意: teacher_logits 是按 shuffle=False 顺序存储的，
          所以这里 __getitem__ 的顺序必须和预计算时一致。
    """
    def __init__(self, csv_path, split, transform, logits_path):
        self.samples = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Usage"] == split:
                    self.samples.append((row["pixels"], int(row["emotion"])))
        self.transform = transform

        data = torch.load(logits_path, map_location="cpu")
        self.teacher_logits = data["logits"]   # (N, 7)
        assert len(self.teacher_logits) == len(self.samples), \
            f"Logits数量({len(self.teacher_logits)})与样本数({len(self.samples)})不匹配，请重新预计算"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pixels_str, label = self.samples[idx]
        img = np.fromstring(pixels_str, sep=" ").reshape(48, 48).astype(np.uint8)
        img = np.stack([img] * 3, axis=-1)
        img = transforms.ToPILImage()(img)
        if self.transform:
            img = self.transform(img)
        t_logit = self.teacher_logits[idx]   # (7,)
        return img, label, t_logit

class FER2013Val(Dataset):
    """验证集: 只返回 (img, label)，不需要 teacher logits"""
    def __init__(self, csv_path, split, transform):
        self.samples = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Usage"] == split:
                    self.samples.append((row["pixels"], int(row["emotion"])))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pixels_str, label = self.samples[idx]
        img = np.fromstring(pixels_str, sep=" ").reshape(48, 48).astype(np.uint8)
        img = np.stack([img] * 3, axis=-1)
        img = transforms.ToPILImage()(img)
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

train_dataset = FER2013WithLogits(CSV_PATH, "Training",   train_transform, TRAIN_LOGITS_PATH)
val_dataset   = FER2013Val(CSV_PATH,        "PublicTest", val_transform)

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   BATCH_SIZE, shuffle=False, num_workers=0)

# ===================== SE + CBAM + ASPP 模块 =====================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class CBAM(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//r, 1), nn.ReLU(),
            nn.Conv2d(c//r, c, 1), nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x * self.ca(x)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa = torch.cat([max_pool, avg_pool], dim=1)
        x = x * self.sa(sa)
        return x

class ASPP(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.c1   = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.c2   = nn.Conv2d(in_c, out_c, 3, padding=6,  dilation=6,  bias=False)
        self.c3   = nn.Conv2d(in_c, out_c, 3, padding=12, dilation=12, bias=False)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_c, out_c, 1))
        self.bn   = nn.BatchNorm2d(out_c * 4)
        self.out  = nn.Conv2d(out_c * 4, out_c, 1, bias=False)
    def forward(self, x):
        h, w = x.shape[2:]
        y1 = self.c1(x)
        y2 = self.c2(x)
        y3 = self.c3(x)
        y4 = self.pool(x).repeat(1, 1, h, w)
        return self.out(self.bn(torch.cat([y1, y2, y3, y4], dim=1)))

# ===================== 学生模型 =====================
class StudentModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        mob = models.mobilenet_v3_small(weights="DEFAULT")
        self.backbone = mob.features

        self.se1   = SEBlock(16)
        self.cbam1 = CBAM(16)

        self.se2   = SEBlock(24)
        self.cbam2 = CBAM(24)

        self.se3   = SEBlock(40)
        self.cbam3 = CBAM(48)

        self.aspp = ASPP(576, 128)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.backbone[0:2](x)
        x = self.se1(x);  x = self.cbam1(x)

        x = self.backbone[2:4](x)
        x = self.se2(x);  x = self.cbam2(x)

        x = self.backbone[4:7](x)
        x = self.se3(x)

        x = self.backbone[7:9](x)
        x = self.cbam3(x)

        x = self.backbone[9:](x)
        x = self.aspp(x)
        return self.classifier(x)

# ===================== 蒸馏损失 =====================
class DistillLoss(nn.Module):
    def __init__(self, temp=4, focal_weight=0.5, kl_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.temp = temp
        self.focal_weight = focal_weight
        self.kl_weight = kl_weight
    def forward(self, s_logit, t_logit, label):
        hard = self.ce(s_logit, label)
        soft = self.kl(
            torch.log_softmax(s_logit / self.temp, dim=1),
            torch.softmax(t_logit  / self.temp, dim=1)
        ) * (self.temp ** 2)
        return self.focal_weight * hard + self.kl_weight * soft

# ===================== 训练 =====================
model     = StudentModel(NUM_CLASSES).to(DEVICE)
criterion = DistillLoss(temp=4, focal_weight=0.5, kl_weight=0.5)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
best_acc  = 0.0

print(f"✅ 快速蒸馏训练启动 (离线 Logits) | 设备: {DEVICE}")

for ep in range(EPOCHS):
    model.train()
    train_loss, corr, total = 0.0, 0, 0
    for img, lab, t_logit in tqdm(train_loader, desc=f"Epoch {ep+1}/{EPOCHS}"):
        img, lab   = img.to(DEVICE), lab.to(DEVICE)
        t_logit    = t_logit.to(DEVICE)   # 直接从文件读，无需 teacher 推理
        s_out      = model(img)
        loss       = criterion(s_out, t_logit, lab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * img.size(0)
        corr  += (s_out.argmax(1) == lab).sum().item()
        total += img.size(0)

    train_loss /= total
    train_acc   = 100.0 * corr / total

    model.eval()
    t_corr, t_total = 0, 0
    with torch.no_grad():
        for img, lab in val_loader:
            img, lab = img.to(DEVICE), lab.to(DEVICE)
            t_corr  += (model(img).argmax(1) == lab).sum().item()
            t_total += img.size(0)
    val_acc = 100.0 * t_corr / t_total

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_student.pth")
        marker = f" ✅ 最佳模型保存 | Val Acc: {best_acc:.2f}%"
    else:
        marker = ""

    print(f"Epoch {ep+1} | Loss {train_loss:.4f} | Train {train_acc:.2f}% | Val {val_acc:.2f}% | Best {best_acc:.2f}%{marker}")
    scheduler.step()

print(f"\n训练完成！最佳验证准确率: {best_acc:.2f}%")
