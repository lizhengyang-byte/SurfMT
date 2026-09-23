"""SurfMT-RandomForest tree-model professional analysis entry point.

Trains the per-task RandomForestRegressor models (retraining at analysis time —
not persisted), then produces a full interpretability bundle under
``outputs/rf_analysis/``:

  1. Feature-importance suite — native gain/split, block aggregation,
     permutation importance (whole-block and per-descriptor).
  2. SHAP interpretability — per-task beeswarm, dependence, waterfall,
     global block |SHAP|.
  3. Fingerprint bit -> Morgan substructure reverse mapping for the top
     most-important bits.
  4. A structured Chinese markdown report.

Health note on tree-scale: Random Forest is scale-invariant on the feature
side, so standardization of the *target* does not affect any of this analysis —
features are used at their raw scale throughout. Unlike the LightGBM baseline,
sklearn's RandomForestRegressor never standardizes the target internally; only
an optional per-task log-transform is applied (see ``train.py``), which is
inverted here before metric computation.

Usage:
    python surfmt_rf/analyze.py                        # all tasks
    python surfmt_rf/analyze.py --task 0               # single task only
    python surfmt_rf/analyze.py --output_dir outputs/rf_analysis --seed 42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running as `python surfmt_rf/analyze.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from surfmt_rf.data import load_data
from surfmt_rf.train import train_all_tasks, N_ESTIMATORS_CANDIDATES
from surfmt_rf.metrics import compute_metrics, TASK_NAMES
from surfmt_rf import interpret as itp


def _denorm(pred_model, transforms):
    """Invert the per-task target transform ('log' | 'none') to raw scale."""
    denorm = np.zeros_like(pred_model)
    for t, transform in enumerate(transforms):
        if transform == "log":
            denorm[:, t] = np.exp(pred_model[:, t])
        else:
            denorm[:, t] = pred_model[:, t]
    return denorm


def main():
    parser = argparse.ArgumentParser(description="SurfMT-RandomForest tree analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/rf_analysis")
    parser.add_argument(
        "--candidates", type=str, default=None,
        help="Comma-separated n_estimators sweep, e.g. '100,300,600,1000'. "
             "Defaults to train.N_ESTIMATORS_CANDIDATES.",
    )
    parser.add_argument("--shap_samples", type=int, default=500,
                        help="Max rows subsampled per task for SHAP.")
    parser.add_argument("--top_bits", type=int, default=20,
                        help="Number of top fingerprint bits per block to map to substructures.")
    parser.add_argument("--top_desc", type=int, default=10,
                        help="Top descriptors per task reported.")
    parser.add_argument("--task", type=int, default=None,
                        help="Analyze a single task index (0-5); default all.")
    parser.add_argument("--no_shap", action="store_true",
                        help="Skip the SHAP stage (native importance only).")
    args = parser.parse_args()

    candidates = N_ESTIMATORS_CANDIDATES
    if args.candidates:
        candidates = sorted({int(c) for c in args.candidates.split(",")})

    out_dir = Path(args.output_dir)
    fi_dir = out_dir / "feature_importance"
    shap_dir = out_dir / "shap"
    sub_dir = out_dir / "substructures"
    for d in (fi_dir, shap_dir, sub_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("SurfMT-RandomForest 树模型专业分析")
    print(f"seed={args.seed}  output_dir={out_dir}")
    print("=" * 64)

    # ---- Load data & train per-task models (retrain at analysis time) ----
    print("\n[1/5] 加载数据 + 重训每任务模型 ...")
    train, test = load_data()
    n_train, n_features = train.X.shape
    n_test = test.X.shape[0]
    models, logs, transforms = train_all_tasks(
        train.X, train.y, train.mask, train.fold,
        test.X, test.y, test.mask,
        params=None, seed=args.seed, candidates=candidates,
    )
    tasks = list(range(len(models))) if args.task is None else [args.task]
    feature_names = itp.build_feature_names()

    # Test metrics (denormalized) for report context.
    pred_model = np.stack([m.predict(test.X) for m in models], axis=1)
    denorm = _denorm(pred_model, transforms)
    test_metrics = compute_metrics(denorm, test.y, test.mask, TASK_NAMES)

    # ------------------------------------------------------------------ #
    # Stage 1: Feature-importance suite
    # ------------------------------------------------------------------ #
    print("\n[2/5] 特征重要性全套 (gain/split/块/置换) ...")
    block_gain = {}
    block_perm = {}
    desc_gain_matrix = np.zeros((len(models), len(itp.NAMED_COLS)))
    top_desc = {}
    for t in tasks:
        model = models[t]
        tsel = test.mask[:, t].astype(bool)

        gain = itp.gain_importance(model)
        split = itp.split_importance(model)
        bf = itp.block_importance(gain)
        block_gain[TASK_NAMES[t]] = bf

        # Whole-block permutation importance on the task's test (labeled) rows.
        imp_perm, base_rmse = itp.block_permutation_importance(
            model, test.X[tsel], test.y[tsel, t])
        block_perm[TASK_NAMES[t]] = imp_perm

        # Record per-descriptor gain (normalized within-task) for heatmap/report.
        g_named = gain[itp.NAMED_COLS]
        norm = g_named / gain.sum() if gain.sum() > 1e-12 else g_named
        desc_gain_matrix[t] = norm

        # Save top-50 gain + split JSON per task.
        order = np.argsort(gain)[::-1][:50]
        fi_dir.joinpath(f"gain_{t}.json").write_text(json.dumps(
            [{"feature": feature_names[i], "importance": float(gain[i])}
             for i in order], indent=2), encoding="utf-8")
        fi_dir.joinpath(f"split_{t}.json").write_text(json.dumps(
            [{"feature": feature_names[i], "importance": float(split[i])}
             for i in np.argsort(split)[::-1][:50]], indent=2), encoding="utf-8")

        # Per-descriptor permutation importance (named cols) -> for report.
        desc_perm, _ = itp.descriptor_permutation_importance(
            model, test.X[tsel], test.y[tsel, t])
        fi_dir.joinpath(f"descriptor_perm_{t}.json").write_text(
            json.dumps(desc_perm, indent=2), encoding="utf-8")

        # Gain-based top descriptors (overwritten with richer SHAP numbers below).
        order = np.argsort(norm)[::-1][:args.top_desc]
        top_desc[TASK_NAMES[t]] = [(itp.DESC_COL_NAMES[i], float(norm[i]), float("nan"))
                                   for i in order]

    itp.plot_block_gain_importance(block_gain, fi_dir / "block_gain_importance.png")
    itp.plot_permutation_block(block_perm, fi_dir / "block_permutation_importance.png")
    itp.plot_descriptor_importance_heatmap(desc_gain_matrix, fi_dir / "descriptor_importance_heatmap.png")
    print("  特征重要性图已生成:", fi_dir)

    # ------------------------------------------------------------------ #
    # Stage 2: SHAP interpretability
    # ------------------------------------------------------------------ #
    block_shap = {}
    shap_insights = []
    if args.no_shap:
        print("\n[3/5] 跳过 SHAP（--no_shap）")
        shap_insights.append("- 已使用 `--no_shap` 跳过 SHAP 阶段，仅保留原生重要性。")
    else:
        print(f"\n[3/5] SHAP 解释性（每任务子采样 ≤{args.shap_samples} 行）...")
        rng = np.random.default_rng(args.seed)
        all_shap_blocks = {}
        for t in tasks:
            model = models[t]
            sel = train.mask[:, t].astype(bool)
            X_t = train.X[sel]
            # Deterministic subsample for SHAP.
            if len(X_t) > args.shap_samples:
                idx = rng.choice(len(X_t), size=args.shap_samples, replace=False)
            else:
                idx = np.arange(len(X_t))
            X_sub = X_t[idx]
            _, sv = itp.shap_values(model, X_sub)

            bs = itp.block_shap_importance(sv)
            block_shap[TASK_NAMES[t]] = bs
            all_shap_blocks[f"task_{t}"] = sv

            # Descriptor/temp SHAP subset.
            sv_desc = sv[:, itp.NAMED_COLS]
            x_desc = X_sub[:, itp.NAMED_COLS]
            mean_abs_desc = np.mean(np.abs(sv_desc), axis=0)

            itp.plot_shap_summary(sv_desc, x_desc,
                                  shap_dir / f"shap_summary_{TASK_NAMES[t]}.png")
            # Dependence for the top-3 descriptors.
            top3 = np.argsort(mean_abs_desc)[::-1][:3]
            for j in top3:
                itp.plot_shap_dependence(int(j), sv_desc, x_desc,
                                         shap_dir / f"shap_dependence_{TASK_NAMES[t]}_{itp.DESC_COL_NAMES[j]}.png")
            # Waterfall for the most-attributed representative sample.
            rep = int(np.argmax(np.abs(sv_desc).sum(axis=1)))
            itp.plot_shap_waterfall(sv_desc, x_desc, rep,
                                    shap_dir / f"shap_waterfall_{TASK_NAMES[t]}.png")

            # Merge with gain into the report's top-descriptor table.
            g_named = itp.gain_importance(model)[itp.NAMED_COLS]
            g_total = itp.gain_importance(model).sum()
            order = np.argsort(mean_abs_desc)[::-1][:args.top_desc]
            top_desc[TASK_NAMES[t]] = [
                (itp.DESC_COL_NAMES[i], float(g_named[i] / g_total),
                 float(mean_abs_desc[i]))
                for i in order]

            ins = (f"- **{TASK_NAMES[t]}**：SHAP 平均 |贡献| 最高的描述符为 "
                   f"`{itp.DESC_COL_NAMES[order[0]]}`（{mean_abs_desc[order[0]]:.4f}）。")
            shap_insights.append(ins)

        itp.plot_shap_block_importance(block_shap, shap_dir / "shap_block_importance.png")
        # Persist raw SHAP matrices as a single .npz.
        np.savez(shap_dir / "shap_values.npz", **all_shap_blocks)
        print("  SHAP 图已生成:", shap_dir)

    # ------------------------------------------------------------------ #
    # Stage 3: Fingerprint bit -> substructure mapping
    # ------------------------------------------------------------------ #
    print("\n[4/5] 指纹位 → 子结构反向映射 ...")
    # Aggregate raw gain across analyzed tasks for ecfp4/ecfp6 bits.
    agg = {}
    for t in tasks:
        gain = itp.gain_importance(models[t])
        for block in ("ecfp4", "ecfp6"):
            a, b = itp.FEATURE_BLOCKS[block]
            arr = agg.setdefault(block, np.zeros(b - a))
            arr += gain[a:b]
    # Radius is 2 (ECFP4) / 3 (ECFP6).
    block_radius = {"ecfp4": 2, "ecfp6": 3}
    substructures = []
    for block in ("ecfp4", "ecfp6"):
        arr = agg[block]
        top_local = np.argsort(arr)[::-1][:args.top_bits]
        for lb in top_local:
            frags = itp.bit_substructures(train.smiles, int(lb), block_radius[block])
            substructures.append({
                "block": block, "global_index": int(itp.FEATURE_BLOCKS[block][0] + lb),
                "local_bit": int(lb), "radius": block_radius[block],
                "importance": float(arr[lb]), "fragments": frags,
            })
    sub_dir.joinpath("top_fingerprint_bits.json").write_text(
        json.dumps(substructures, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  已映射", len(substructures), "个指纹位 ->", sub_dir / "top_fingerprint_bits.json")

    # ------------------------------------------------------------------ #
    # Stage 4: Markdown report
    # ------------------------------------------------------------------ #
    print("\n[5/5] 生成 Markdown 分析报告 ...")
    meta = {
        "seed": args.seed, "n_train": int(n_train), "n_test": int(n_test),
        "n_features": int(n_features), "candidates": list(candidates),
        "shap_samples": args.shap_samples, "top_bits": args.top_bits,
        "logs": {f"task_{t}": logs[f"task_{t}"] for t in tasks},
        "block_gain": block_gain, "block_perm": block_perm,
        "top_desc": top_desc, "substructures": substructures[:40],
        "shap_insights": shap_insights,
        "test_metrics": {k: (None if isinstance(v, float) and np.isnan(v) else float(v))
                         for k, v in test_metrics.items()},
    }
    report_path = itp.write_report(out_dir / "report.md", meta)
    print("  报告已生成:", report_path)

    # ---- Summary ------------------------------------------------------- #
    summary = {
        "seed": args.seed,
        "tasks": tasks,
        "n_train": int(n_train), "n_features": int(n_features),
        "test_metrics": meta["test_metrics"],
        "output_dir": str(out_dir),
        "files": {
            "feature_importance": [str(fi_dir / f) for f in
                                   ("block_gain_importance.png", "block_permutation_importance.png",
                                    "descriptor_importance_heatmap.png")],
            "shap": [str(shap_dir / f) for f in ("shap_block_importance.png",)],
            "substructures": str(sub_dir / "top_fingerprint_bits.json"),
            "report": str(report_path),
        },
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 64)
    print("分析完成。产物位于:", out_dir)
    print("  - 报告: report.md")
    print("  - 摘要: analysis_summary.json")
    print("=" * 64)
    return summary


if __name__ == "__main__":
    main()