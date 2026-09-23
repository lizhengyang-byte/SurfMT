"""Tree-model interpretability helpers for SurfMT-RandomForest.

Provides the building blocks used by ``analyze.py``: feature-name mapping,
Random Forest native / block / permutation feature importance, SHAP
interpretability, Morgan bit -> substructure reverse mapping, plotting, and
report generation.

The model is a set of per-task ``RandomForestRegressor`` (see ``train.py``),
each trained on a 4329-dim tabular vector (ECFP4 + ECFP6 + MACCS + 65
descriptors + temperature). Only the 65 descriptor columns and the temperature
column have human-readable names; the ~4263 fingerprint/MACCS bits are labelled
``<block>_<index>``. See ``features.py`` for block offsets.

Health note vs. the LightGBM analysis: the LightGBM ``analyze.py`` reports a
`native gain` importance (``feature_importance(importance_type='gain')``) and a
`split` importance (raw split tally from the booster). Scikit-learn's
``RandomForestRegressor`` exposes its impurity-based ``feature_importances_``,
which is the direct Random Forest analogue of gain; there is no built-in split
count, so we tally split occurrences across every tree in ``estimators_`` (each
``sklearn.tree`` stores the feature used at every node, -2 = leaf). Both are
reported here so the two tree baselines offer the same surface.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering; must run before pyplot is imported
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import FindAtomEnvironmentOfRadiusN, PathToSubmol

from .features import (
    FEATURE_DIM,
    DESC_N,
    TEMP_N,
    ECFP4_BITS,
    ECFP6_BITS,
    MACCS_BITS,
    _SLICES,
    DESCRIPTOR_NAMES,
)
from .metrics import TASK_NAMES

# --------------------------------------------------------------------------- #
# Feature / block naming
# --------------------------------------------------------------------------- #

FEATURE_BLOCKS = {name: tuple(slc) for name, slc in _SLICES.items()}


def build_feature_names():
    """Return 4329 human-readable feature names aligned with the feature matrix.

    Blocks are ``ecfp4_*``, ``ecfp6_*``, ``maccs_*`` (all indexed bits),
    followed by the 65 RDKit descriptor names and finally ``temp``.
    """
    names = [f"ecfp4_{i}" for i in range(ECFP4_BITS)]
    names += [f"ecfp6_{i}" for i in range(ECFP6_BITS)]
    names += [f"maccs_{i}" for i in range(MACCS_BITS)]
    names += list(DESCRIPTOR_NAMES)
    names.append("temp")
    assert len(names) == FEATURE_DIM, f"expected {FEATURE_DIM}, got {len(names)}"
    return names


# Indices of the interpretable (named) columns: 65 descriptors + temperature.
NAMED_COLS = list(range(_SLICES["desc"][0], _SLICES["desc"][1])) + [FEATURE_DIM - 1]
DESC_COL_NAMES = list(DESCRIPTOR_NAMES) + ["temp"]


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #

def gain_importance(model):
    """Random Forest impurity-based importance (MDA, mean decrease in impurity).

    Equivalent to ``model.feature_importances_`` and the Random Forest analogue
    of LightGBM's raw gain importance -> [4329].
    """
    return np.asarray(model.feature_importances_, dtype=float)


def split_importance(model):
    """Split (frequency) importance for a Random Forest -> [4329].

    Counts every split decision made by every tree in ``estimators_``. In
    ``sklearn.tree`` each node stores the feature index it splits on (leaf
    nodes are ``-2``), so a simple tally gives the same *how-often-is-a-feature
    used* statistic that LightGBM reports as `split` importance.
    """
    imp = np.zeros(model.n_features_in_, dtype=float)
    for est in model.estimators_:
        features = est.tree_.feature
        counts = np.bincount(features[features >= 0], minlength=model.n_features_in_)
        imp += counts
    return imp


def block_importance(imp_vec, normalize=True):
    """Sum an importance vector per feature block.

    If ``normalize``, each block is expressed as a fraction of total importance
    so values are comparable across tasks with different magnitudes.
    """
    out = {name: float(imp_vec[start:end].sum())
           for name, (start, end) in FEATURE_BLOCKS.items()}
    total = float(imp_vec.sum())
    if normalize and total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def _rmse(y_true, y_pred):
    """RMSE — used for permutation importance.

    R² is unstable for permutation importance on small tasks whose reference-set
    variance is tiny (a tiny ss_tot blows the R² drop up to ~1e10). RMSE is
    always positive and finite, so it is the safer drop-metric here.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def block_permutation_importance(model, X, y, n_perm=5, seed=42):
    """Reference-set RMSE increase when each whole feature block is shuffled.

    Computed on ``X``/``y`` (a labeled row subset, e.g. the task's test rows)
    so it is a honest generalization estimate rather than in-sample.
    Returns (imp_dict {block: rmse_increase}, base_rmse).
    """
    rng = np.random.default_rng(seed)
    base = _rmse(y, model.predict(X))
    imp = {}
    for block, (a, b) in FEATURE_BLOCKS.items():
        rises = []
        for _ in range(n_perm):
            Xp = X.copy()
            for c in range(a, b):
                rng.shuffle(Xp[:, c])
            rises.append(_rmse(y, model.predict(Xp)) - base)
        imp[block] = float(np.mean(rises))
    return imp, float(base)


def descriptor_permutation_importance(model, X, y, n_perm=5, seed=42):
    """Per-descriptor (+temp) RMSE-increase permutation importance on named cols."""
    rng = np.random.default_rng(seed)
    base = _rmse(y, model.predict(X))
    imp = {}
    for c, name in zip(NAMED_COLS, DESC_COL_NAMES):
        rises = []
        for _ in range(n_perm):
            Xp = X.copy()
            rng.shuffle(Xp[:, c])
            rises.append(_rmse(y, model.predict(Xp)) - base)
        imp[name] = float(np.mean(rises))
    return imp, float(base)


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #

def shap_values(model, X):
    """Tree SHAP values for a Random Forest on ``X`` -> [N, 4329].

    ``shap.TreeExplainer`` supports ``RandomForestRegressor`` natively; callers
    should sub-sample ``X`` first (see ``analyze.py --shap_samples``).
    """
    explainer = shap.TreeExplainer(model)
    return explainer, np.asarray(explainer.shap_values(X), dtype=float)


def block_shap_importance(shap_mat):
    """Mean |SHAP| per feature block (averaged over rows)."""
    mean_abs = np.mean(np.abs(shap_mat), axis=0)
    return {name: float(mean_abs[start:end].sum())
            for name, (start, end) in FEATURE_BLOCKS.items()}


# --------------------------------------------------------------------------- #
# Morgan bit -> substructure reverse mapping
# --------------------------------------------------------------------------- #

def bit_substructures(smiles_list, local_bit, radius, max_samples=60):
    """Return the distinct sub-structures that set one hashed Morgan bit.

    ``local_bit`` is the bit index within the ECFP4 (radius=2) or ECFP6
    (radius=3) 2048-bit block. We re-derive each molecule's bit-info via the
    legacy ``GetMorganFingerprintAsBitVect`` (same default invariants/radius as
    ``features.py``), then for every ``(atomIdx, r)`` environment that hashed
    to that bit reconstruct the surrounding fragment with
    ``FindAtomEnvironmentOfRadiusN`` + ``PathToSubmol``.

    Note the fragment is the *representative* circular environment; because
    Morgan bits are hashed, several distinct environments can collide onto the
    same bit, so treat the returned fragments as the most common meanings.
    """
    envs = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        info = {}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")          # Python warnings
                RDLogger.DisableLog("rdApp.warning")     # RDKit C++ logger
                rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, radius=radius, nBits=2048, bitInfo=info)
        except Exception:
            continue
        if local_bit not in info:
            continue
        for atom_idx, r in info[local_bit]:
            try:
                env = FindAtomEnvironmentOfRadiusN(mol, r, atom_idx)
                amap = {}
                sub = PathToSubmol(mol, env, atomMap=amap)
                envs.append(Chem.MolToSmiles(sub))
            except Exception:
                continue
        if len(envs) >= max_samples:
            break
    # De-duplicate while keeping the most common order.
    return list(dict.fromkeys(envs))


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

_PALETTE = sns.color_palette("deep")


def _style():
    sns.set_theme(style="whitegrid", context="notebook")
    # CJK-capable font so Chinese axis labels render (drops to a fallback if
    # "Microsoft YaHei" / "SimHei" are absent, e.g. on Linux).
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"


def plot_block_gain_importance(block_fractions_per_task, out_path):
    """Grouped horizontal bar: normalized per-block gain across tasks."""
    _style()
    blocks = list(FEATURE_BLOCKS.keys())
    x = np.arange(len(blocks))
    w = 0.8 / max(1, len(TASK_NAMES))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (task, fractions) in enumerate(block_fractions_per_task.items()):
        vals = [fractions.get(b, 0.0) for b in blocks]
        ax.bar(x + i * w, vals, w, label=task, color=_PALETTE[i % len(_PALETTE)])
    ax.set_xticks(x + w * (len(block_fractions_per_task) - 1) / 2)
    ax.set_xticklabels(blocks)
    ax.set_ylabel("归一化 gain 占比")
    ax.set_title("特征分块 gain 重要性（按任务归一化）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_block_importance_generic(block_vals, out_path, title, ylabel="置换 R² 下降",
                                  sort_desc=True):
    _style()
    items = sorted(block_vals.items(), key=lambda kv: kv[1], reverse=sort_desc)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(names[::-1], vals[::-1], color=_PALETTE[0])
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_permutation_block(block_perm_tasks, out_path):
    """Grouped bar of block permutation RMSE-increase per task."""
    _style()
    blocks = list(FEATURE_BLOCKS.keys())
    x = np.arange(len(blocks))
    w = 0.8 / max(1, len(block_perm_tasks))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (task, imp) in enumerate(block_perm_tasks.items()):
        vals = [imp.get(b, 0.0) for b in blocks]
        ax.bar(x + i * w, vals, w, label=task, color=_PALETTE[i % len(_PALETTE)])
    ax.set_xticks(x + w * (len(block_perm_tasks) - 1) / 2)
    ax.set_xticklabels(blocks)
    ax.set_ylabel("置换 RMSE 上升")
    ax.set_title("特征分块置换重要性（参考集 RMSE 上升）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_descriptor_importance_heatmap(desc_gain_matrix, out_path, top_n=20):
    """Heatmap of per-task gain importance for the top-N descriptors."""
    _style()
    # desc_gain_matrix: shape [n_tasks, DESC_N+1(temp)] over NAMED_COLS
    ms = np.abs(desc_gain_matrix)
    order = np.argsort(ms.max(axis=0))[::-1][:top_n]
    sel = desc_gain_matrix[:, order]
    labels = [DESC_COL_NAMES[i] for i in order]
    fig, ax = plt.subplots(figsize=(max(5, top_n * 0.55), 0.45 * sel.shape[0] + 1.2))
    sns.heatmap(sel, ax=ax, cmap="viridis", xticklabels=labels,
                yticklabels=TASK_NAMES, cbar_kws={"label": "归一化 gain"})
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"每任务 top-{top_n} 描述符 gain 重要性")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_shap_summary(shap_desc, x_desc, out_path, max_display=15):
    """Beeswarm summary for the descriptor+temp SHAP subset."""
    _style()
    shap.summary_plot(
        shap_desc, x_desc, feature_names=DESC_COL_NAMES,
        max_display=max_display, show=False)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_shap_dependence(feature_index, shap_desc, x_desc, out_path):
    """SHAP dependence for one named column (all interactions)."""
    _style()
    shap.dependence_plot(
        feature_index, shap_desc, x_desc,
        feature_names=DESC_COL_NAMES, interaction_index="auto", show=False)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_shap_waterfall(shap_desc, x_desc, sample_idx, out_path, top_n=12):
    """Manual waterfall for one representative molecule (robust to shap API churn).

    Sorts the descriptor/temp SHAP contributions and renders them as a
    horizontal diverging bar — the SHAP philosophy of starting from an expected
    value. Uses values only, no fragile ``shap.plots.waterfall`` object types.
    """
    _style()
    sv = shap_desc[sample_idx]
    xs = x_desc[sample_idx]
    idx_sorted = np.argsort(np.abs(sv))[::-1][:top_n]
    contribs = sv[idx_sorted]
    names = [DESC_COL_NAMES[i] for i in idx_sorted]
    colors = [_PALETTE[1] if c >= 0 else _PALETTE[0] for c in contribs]
    fig, ax = plt.subplots(figsize=(7, 0.4 * top_n + 1.0))
    ax.barh(names[::-1], contribs[::-1], color=colors[::-1])
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("SHAP 贡献值")
    ax.set_title(f"代表样本 #{sample_idx} 的 SHAP 贡献")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_shap_block_importance(block_shap_tasks, out_path):
    """Grouped bar of mean-|SHAP| per block across tasks."""
    _style()
    blocks = list(FEATURE_BLOCKS.keys())
    x = np.arange(len(blocks))
    w = 0.8 / max(1, len(block_shap_tasks))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (task, imp) in enumerate(block_shap_tasks.items()):
        vals = [imp.get(b, 0.0) for b in blocks]
        ax.bar(x + i * w, vals, w, label=task, color=_PALETTE[i % len(_PALETTE)])
    ax.set_xticks(x + w * (len(block_shap_tasks) - 1) / 2)
    ax.set_xticklabels(blocks)
    ax.set_ylabel("平均 |SHAP|")
    ax.set_title("特征分块 SHAP 重要性")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _md_table(headers, rows):
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    head = "| " + " | ".join(headers) + " |"
    body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([head, sep] + body)


def _fmt(v, nd=4):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def write_report(out_path, meta):
    """Write the Chinese markdown analysis report.

    ``meta`` is a dict assembled by ``analyze.py`` containing: overview, per
    task logs, block importance tables, top descriptors, SHAP insights,
    substructure table, reproducibility and limitations.
    """
    lines = ["# SurfMT-RandomForest 树模型专业分析报告", ""]

    # 1. Overview ------------------------------------------------------------
    lines += ["## 1. 数据集与任务总览", ""]
    lines += [f"- 训练样本数：**{meta['n_train']}**；测试样本数：**{meta['n_test']}**"]
    lines += [f"- 特征维度：**{meta['n_features']}**（ECFP4 {ECFP4_BITS} + ECFP6 "
              f"{ECFP6_BITS} + MACCS {MACCS_BITS} + 描述符 {DESC_N} + 温度 {TEMP_N}）"]
    lines += ["- 每任务为独立 `RandomForestRegressor`，目标经 log 变换（如适用）；"
              "特征保持原始尺度。树棵数经 10 折 CV 选取后在全量标签样本上重训。", ""]
    rows = [[TASK_NAMES[int(k.rsplit("_", 1)[1])], m["samples"],
             m["n_estimators"], m["transform"]]
            for k, m in meta["logs"].items()]
    lines += [_md_table(["任务", "样本数", "n_estimators", "目标变换"], rows), ""]

    # 2. Block importance -----------------------------------------------------
    lines += ["## 2. 特征分块重要性（归一化增益）", ""]
    present_tasks = [t for t in TASK_NAMES if t in meta.get("block_gain", {})]
    rows = [[b] + [meta["block_gain"][t].get(b, 0.0) for t in present_tasks]
            for b in FEATURE_BLOCKS]
    lines += [_md_table(["分块"] + [f"{t}" for t in present_tasks], rows), ""]
    lines += [f"温度近乎常数（多数样本取 25.0），通常贡献极低；描述符与指纹位才是主要信号。", ""]

    lines += ["### 2.1 分块置换重要性（参考集 RMSE 上升）", ""]
    rows = [[b] + [meta["block_perm"][t].get(b, 0.0) for t in present_tasks]
            for b in FEATURE_BLOCKS]
    lines += [_md_table(["分块"] + [f"{t}" for t in present_tasks], rows), ""]

    # 3. Top descriptors per task --------------------------------------------
    lines += ["## 3. 每任务 top 描述符（增益与 SHAP）", ""]
    for t in present_tasks:
        lines += [f"### 任务 `{t}`", ""]
        rows = [[n, _fmt(g), _fmt(s)] for n, g, s in meta["top_desc"].get(t, [])]
        if not rows:
            rows = [["（该任务未在此次分析中包含）", "", ""]]
        lines += [_md_table(["描述符", "归一化 gain", "平均 |SHAP|"], rows), ""]

    # 4. SHAP insights --------------------------------------------------------
    lines += ["## 4. SHAP 洞察", ""]
    if meta["shap_insights"]:
        lines += meta["shap_insights"]
        lines += [""]

    # 5. Substructure mapping --------------------------------------------------
    lines += ["## 5. 关键指纹位 → 化学子结构", ""]
    lines += [f"top-{meta.get('top_bits', 20)} 个高重要性指纹位（按跨任务增益聚合）"
              f"反向映射到 Morgan 圆形环境片段：", ""]
    if meta["substructures"]:
        # Keep the table compact: block / bit index / importance; the full
        # fragment SMILES for each bit live in the JSON beside this report.
        rows = [[r["block"], str(r["local_bit"]), _fmt(r["importance"], 5)]
                for r in meta["substructures"][:20]]
        lines += [_md_table(["指纹块", "位索引", "跨任务重要性"], rows), ""]
        lines += ["（各位的具体片段 SMILES 见 "
                  "`substructures/top_fingerprint_bits.json`）", ""]
    lines += ["**注意**：ECFP 位为哈希位，多个环境可能碰撞到同一位置，"
              "片段是代表性环境而非精确独有结构，须结合分子上下文解读。", ""]

    # 6. Reproducibility ------------------------------------------------------
    lines += ["## 6. 可复现性", ""]
    lines += [f"- 随机种子：`{meta['seed']}`；`shap_samples={meta['shap_samples']}`；"
              f"每任务 `n_estimators` 见上表（CV 从候选 {meta.get('candidates', '…')} 选取）", ""]
    lines += ["- 训练与特征代码：`surfmt_rf/{train,features,data}.py`（未改动，分析时重训）", ""]
    lines += ["- 产物清单见 `analysis_summary.json`", ""]

    # 7. Limitations ----------------------------------------------------------
    lines += ["## 7. 局限", ""]
    lines += [
        "- **哈希碰撞**：2048 位 Morgan 指纹为哈希表示，位到子结构为近似映射。",
        "- **稀疏特征**：指纹位多为 0/1 且极稀疏，个别位的 gain/SHAP 易受样本量影响。",
        "- **数据量小**：各任务标签样本量 42%~90% 不等，SHAP 基于 ≤500 样本子采样。",
        "- **树模型视角**：与 GNN（SurfMT-GNN）的表示不同，本分析仅反映 Random Forest 学到的规律。",
        "",
    ]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)