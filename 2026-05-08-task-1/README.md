# FER System - Facial Expression Recognition

基于深度学习的实时面部表情识别系统，采用知识蒸馏策略实现轻量化高精度推理。

## 项目概述

- **教师模型**: ConvNeXt-Base (微调) — 强特征表征能力
- **学生模型**: MobileNetV3-Small + SE + CBAM + ASPP — 轻量化实时推理
- **训练策略**: 两阶段训练，先训练教师模型再蒸馏训练学生模型
- **推理方式**: OpenCV Haar级联人脸检测 + PyTorch模型推理
- **GUI**: PyQt5实时界面，多线程视频流处理

## 环境安装

```bash
# 创建虚拟环境 (推荐 Python 3.8+)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# 安装依赖
pip install -r requirements.txt
```

## 数据集准备

下载 [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) 数据集并放置到 `data/fer2013/` 目录下：

```bash
# 方式1: 使用下载脚本 (需要kaggle API配置)
python scripts/download_data.py --output-dir data/fer2013

# 方式2: 手动下载并验证
python scripts/download_data.py --local-path data/fer2013/fer2013.csv
```

目录结构应为：
```
data/fer2013/fer2013.csv
```

## 训练

### 阶段1: 训练教师模型

```bash
python scripts/train_teacher.py \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-4 \
    --data-dir data/fer2013 \
    --checkpoint-dir checkpoints/teacher \
    --log-dir logs/teacher
```

### 阶段2: 蒸馏训练学生模型

```bash
python scripts/train_student.py \
    --teacher-path checkpoints/teacher/best_model.pth \
    --epochs 100 \
    --batch-size 64 \
    --lr 3e-4 \
    --temperature 4.0 \
    --data-dir data/fer2013 \
    --checkpoint-dir checkpoints/student \
    --log-dir logs/student
```

## 评估

```bash
python scripts/evaluate.py \
    --model-path checkpoints/student/best_model.pth \
    --model-type student \
    --data-dir data/fer2013 \
    --output reports/eval_report.json
```

## 实时推理 (GUI)

```bash
python main.py --model-path checkpoints/student/best_model.pth --device cpu
```

## 批量推理

```bash
python scripts/batch_infer.py \
    --image-dir path/to/images \
    --model-path checkpoints/student/best_model.pth \
    --output results.csv
```

## 模型导出

```bash
python scripts/export_model.py \
    --model-path checkpoints/student/best_model.pth \
    --output-path checkpoints/student/student_traced.pt \
    --benchmark
```

## 技术架构

### 学生模型结构

```
输入 [B, 3, 48, 48]
  → MobileNetV3-Small Backbone
    → SE注意力 (Layer 0-3)
    → CBAM注意力 (Layer 4-11)
  → ASPP多尺度模块 (dilation=1,6,12,18 + 全局池化)
  → 轻量化分类头 (GAP → Dropout → Linear → Hardswish → Dropout → Linear(7))
输出 [B, 7]
```

### 损失函数

```
L = 0.7 × FocalLoss + 0.3 × KL_Distillation(T=4)
```

### 多线程架构

```
VideoThread (采集) → 主线程 (渲染) → InferenceThread (推理) → 主线程 (更新UI)
```

## 7类表情

| 编号 | 表情 |
|------|------|
| 0    | Angry |
| 1    | Disgust |
| 2    | Fear |
| 3    | Happy |
| 4    | Sad |
| 5    | Surprise |
| 6    | Neutral |

## 项目结构

```
fer_system/
├── data/           # 数据加载与预处理
├── models/         # 模型定义 (SE, CBAM, ASPP, Student, Teacher)
├── training/       # 训练循环与损失函数
├── inference/      # 人脸检测与表情预测
├── gui/            # PyQt5图形界面
├── scripts/        # 训练/评估/推理脚本
├── utils/          # 配置与工具
├── checkpoints/    # 模型权重
├── logs/           # 训练日志
└── main.py         # GUI入口
```
