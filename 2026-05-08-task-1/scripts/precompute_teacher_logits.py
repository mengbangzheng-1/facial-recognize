"""
预计算教师模型 Logits 脚本
一次性把所有训练集 + 验证集的 teacher 输出算好存成 .pt 文件
之后训练时直接读，不再每个 batch 都跑 ConvNeXt-Base，速度大幅提升

用法:
    python scripts/precompute_teacher_logits.py

输出:
    data/fer2013/teacher_logits_train.pt   训练集 logits
    data/fer2013/teacher_logits_val.pt     验证集 logits
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import csv
import numpy as np

# ===================== 配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "data/fer2013/fer2013.csv"
TEACHER_CKPT = "model_checkpoints/teacher/best_teacher.pth"
BATCH_SIZE = 256
NUM_CLASSES = 7
IMAGE_SIZE = 224

# ===================== FER2013 数据集 =====================
class FER2013Dataset(Dataset):
    def __init__(self, csv_path, split="Training", transform=None):
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

# 预计算时不做随机增强，保证 logits 稳定
precompute_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===================== 加载教师模型 =====================
print(f"加载教师模型: {TEACHER_CKPT}")
teacher = models.convnext_base(weights=None)
teacher.classifier[-1] = nn.Linear(1024, NUM_CLASSES)
teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location=DEVICE), strict=False)
teacher = teacher.to(DEVICE).eval()
for p in teacher.parameters():
    p.requires_grad = False

print(f"✅ 教师模型加载完成 | 设备: {DEVICE}")

# ===================== 预计算函数 =====================
def precompute(split_name, save_path):
    dataset = FER2013Dataset(CSV_PATH, split_name, precompute_transform)
    loader  = DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=0)

    all_logits = []
    all_labels = []

    print(f"\n开始预计算 [{split_name}] | 样本数: {len(dataset)}")
    with torch.no_grad():
        for img, lab in tqdm(loader, desc=f"预计算 {split_name}"):
            img = img.to(DEVICE)
            logit = teacher(img)
            all_logits.append(logit.cpu())
            all_labels.append(lab)

    all_logits = torch.cat(all_logits, dim=0)  # (N, 7)
    all_labels = torch.cat(all_labels, dim=0)  # (N,)

    torch.save({"logits": all_logits, "labels": all_labels}, save_path)
    print(f"✅ 已保存 [{split_name}] logits -> {save_path}")
    print(f"   logits shape: {all_logits.shape} | labels shape: {all_labels.shape}")

# ===================== 执行 =====================
precompute("Training",   "data/fer2013/teacher_logits_train.pt")
precompute("PublicTest", "data/fer2013/teacher_logits_val.pt")

print("\n✅ 预计算完成！接下来运行 train_distill_fast.py 即可加速训练")
