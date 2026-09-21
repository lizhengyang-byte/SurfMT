# SurfMT-GNN: 多任务图神经网络表面活性剂性质预测

基于论文 *Multi-task graph neural networks for comprehensive surfactant property prediction* (Digital Discovery, 2026) 的 PyTorch 实现。

## 项目概述

SurfMT-GNN 是一个三分支多任务图神经网络，同时预测 6 项表面活性剂关键界面性质：

| 性质 | 符号 | 说明 |
|------|------|------|
| 临界胶束浓度负对数 | pCMC | -log₁₀[CMC/M] |
| CMC 下表面张力 | γ_CMC | mN·m⁻¹ |
| 最大表面过剩 | Γ_max | μmol·m⁻² |
| 最小分子占有面积 | A_min | nm² |
| CMC 下表面压 | π_CMC | mN·m⁻¹ |
| 表面活性剂效率 | pC₂₀ | 降低表面张力 20 mN/m 所需浓度负对数 |

## 模型架构

**三编码分支 → 特征融合 → 共享表征层 → 6 个独立任务输出头**

1. **AttentiveFP 图编码器**：3 层消息传递，256 隐藏维度，4 注意力头，dropout=0.1
2. **温度 MLP 编码器**：归一化温度输入，1→32→64，GELU 激活
3. **分子描述符 MLP 分支**：12 个 RDKit 描述符，Z-score 标准化，12→32→64

融合层将三个分支拼接（256+64+64=384 维）后通过 MLP 降维至 256，再经共享层得到 128 维共享表征，最后通过 6 个并行 MLP 头输出各性质预测。

## 数据

使用 SurfPro 数据集，存放于 `data/surfpro/`：
- `surfpro_train.csv` — 训练集（1335 样本，含 10 折交叉验证划分）
- `surfpro_test.csv` — 测试集（140 样本）

数据存在不同程度缺失，采用**掩码 MSE 损失**处理缺失标签。

## 环境要求

```
Python >= 3.9
PyTorch >= 2.0.0
PyTorch Geometric >= 2.3.0
RDKit >= 2022.09.5
numpy, pandas, scikit-learn, scipy
```

安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 单模型训练与评估

```bash
python scripts/train_single.py --seed 42 --output_dir outputs/single_seed42
```

### 10 折交叉验证

```bash
python scripts/train_cv.py --seed 42 --output_dir outputs/cv_seed42
```

### 完整集成训练（6 种子 × 10 折 = 60 模型）

```bash
python scripts/train_ensemble.py --output_dir outputs/ensemble
```

### 集成评估与不确定性量化

```bash
python scripts/evaluate_ensemble.py --ensemble_dir outputs/ensemble --output_dir outputs/eval
```

## 训练配置

- 优化器：AdamW（lr=5e-4, weight_decay=1e-4）
- 学习率：warmup 10 轮 + 余弦退火热重启
- 梯度裁剪：max_norm=1.0
- Batch size：32
- 最大 epoch：500
- 早停：patience=80
- 任务权重：Γ_max=1.5, γ_CMC=1.3, π_CMC=1.3, A_min=1.1, pCMC=1.0, pC₂₀=1.0

## 项目结构

```
├── data/surfpro/              # 数据集
├── surfmt_gnn/                # 核心包
│   ├── config.py              # 超参数配置
│   ├── data/                  # 数据模块
│   │   ├── featurizer.py      # 原子/键特征编码 (39/10-dim)
│   │   ├── descriptors.py     # RDKit 描述符
│   │   └── dataset.py         # PyG 数据集
│   ├── models/                # 模型模块
│   │   ├── modules.py         # MLP 通用模块
│   │   ├── attentive_fp.py    # 多头 AttentiveFP
│   │   └── surfmt_gnn.py      # 主模型
│   ├── training/              # 训练模块
│   │   ├── loss.py            # 掩码 MSE 损失
│   │   ├── scheduler.py       # LR 调度器
│   │   └── trainer.py         # 训练器
│   ├── evaluation/
│   │   └── metrics.py         # R²/RMSE/MAE
│   └── utils/
│       └── seed.py            # 随机种子
├── scripts/                   # 运行脚本
│   ├── train_single.py
│   ├── train_cv.py
│   ├── train_ensemble.py
│   └── evaluate_ensemble.py
└── requirements.txt
```

## 参考文献

- 论文: *Multi-task graph neural networks for comprehensive surfactant property prediction*, Digital Discovery, 2026
- 原仓库: https://github.com/albakhrani/SurfMT-GNN
- 数据集: SurfPro (Zenodo: 10.5281/zenodo.20761125)
