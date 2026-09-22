# SurfMT: 表面活性剂关键界面性质多任务预测

SurfMT 是一个表面活性剂界面性质预测项目，除主模型 **SurfMT-GNN**（基于论文 *Multi-task graph neural networks for comprehensive surfactant property prediction*, Digital Discovery 2026 的 PyTorch 实现）外，还提供 LightGBM、Random Forest 等逐任务回归基线，便于横向对比。

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

### SurfMT-GNN 单模型训练与评估

```bash
python surfmt_gnn/scripts/train_single.py --seed 42 --output_dir outputs/single_seed42
```

### SurfMT-GNN 10 折交叉验证

```bash
python surfmt_gnn/scripts/train_cv.py --seed 42 --output_dir outputs/cv_seed42
```

### SurfMT-GNN 完整集成训练（6 种子 × 10 折 = 60 模型）

```bash
python surfmt_gnn/scripts/train_ensemble.py --output_dir outputs/ensemble
```

### SurfMT-GNN 集成评估与不确定性量化

```bash
python surfmt_gnn/scripts/evaluate_ensemble.py --ensemble_dir outputs/ensemble --output_dir outputs/eval
```

### 树模型基线（表格特征 + 逐任务回归）

树模型不依赖分子图结构，仅用 SMILES + 温度 + 12 描述符计算出的表格特征，
每个任务单独训练一个回归器，掩码处理缺失标签。

**LightGBM 基线（CV 选取最优 boosting 轮数）：**

```bash
python surfmt_lgb/main.py --seed 42 --output_dir outputs/lgb_seed42
python surfmt_lgb/main.py --hetero --output_dir outputs/lgb_hetero      # 异构集成
```

**Random Forest 基线（CV 选取最优树数）：**

```bash
python surfmt_rf/main.py --seed 42 --output_dir outputs/rf_seed42
python surfmt_rf/main.py --hetero --output_dir outputs/rf_hetero       # 异构集成
```

两种基线都支持 `--seeds 42,123,456` 多种子平均集成。

## SurfMT-GNN 训练配置

- 优化器：AdamW（lr=5e-4, weight_decay=1e-4）
- 学习率：warmup 10 轮 + 余弦退火热重启
- 梯度裁剪：max_norm=1.0
- Batch size：32
- 最大 epoch：500
- 早停：patience=80
- 任务权重：Γ_max=1.5, γ_CMC=1.3, π_CMC=1.3, A_min=1.1, pCMC=1.0, pC₂₀=1.0

## 项目结构

每种模型一个顶层文件夹，自包含全部代码与运行脚本；`outputs/` 为日志与结果存档。

```
├── data/surfpro/              # 数据集（各模型共享）
├── surfmt_gnn/                # SurfMT-GNN（PyTorch 多任务图神经网络）
│   ├── __init__.py
│   ├── config.py              # 超参数配置
│   ├── data/                  # 数据模块
│   │   ├── featurizer.py      # 原子/键特征编码 (39/10-dim)
│   │   ├── descriptors.py     # RDKit 描述符
│   │   ├── fingerprints.py    # Morgan 指纹
│   │   ├── utils.py           # 缩放器工具
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
│   ├── utils/
│   │   └── seed.py            # 随机种子
│   └── scripts/               # 运行脚本
│       ├── train_single.py
│       ├── train_cv.py
│       ├── train_ensemble.py
│       └── evaluate_ensemble.py
├── surfmt_lgb/                # SurfMT-LightGBM 基线（逐任务回归）
│   ├── features.py            # 特征提取 (ECFP4/6 + MACCS + 描述符 + 温度)
│   ├── data.py                # CSV 加载、缺失掩码、折划分
│   ├── train.py               # 逐任务 LightGBM 训练与早停
│   ├── metrics.py             # 掩码 R²/RMSE/MAE
│   └── main.py                # 端到端入口
├── surfmt_rf/                 # SurfMT-RandomForest 基线（逐任务回归）
│   ├── features.py            # 特征提取 (同 surfmt_lgb)
│   ├── data.py                # CSV 加载、缺失掩码、折划分
│   ├── train.py               # 逐任务 RandomForest 训练（CV 选树数）
│   ├── metrics.py             # 掩码 R²/RMSE/MAE
│   └── main.py                # 端到端入口
├── outputs/                   # 日志与结果存档（每个模型输出）
└── requirements.txt
```

## 参考文献

- 论文: *Multi-task graph neural networks for comprehensive surfactant property prediction*, Digital Discovery, 2026
- 原仓库: https://github.com/albakhrani/SurfMT-GNN
- 数据集: SurfPro (Zenodo: 10.5281/zenodo.20761125)
