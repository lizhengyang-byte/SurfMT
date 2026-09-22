"""Train full deep ensemble: 10-fold CV x 6 seeds = 60 models.

Usage:
    python scripts/train_ensemble.py --output_dir outputs/ensemble
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from surfmt_gnn.config import Config


def main():
    parser = argparse.ArgumentParser(description="Train full ensemble (6 seeds x 10 folds)")
    parser.add_argument(
        "--output_dir", type=str, default="outputs/ensemble",
        help="Base output directory"
    )
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seeds (default: all 6 seeds from config)"
    )
    parser.add_argument(
        "--folds", type=str, default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated fold indices"
    )
    args = parser.parse_args()

    config = Config()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else config.seeds

    print("=" * 60)
    print("SurfMT-GNN Ensemble Training")
    print(f"Seeds: {seeds}")
    print(f"Folds: {args.folds}")
    print(f"Total models: {len(seeds)} x {len(args.folds.split(','))} = {len(seeds) * len(args.folds.split(','))}")
    print("=" * 60)

    all_results = {}

    for seed in seeds:
        print(f"\n{'#' * 60}")
        print(f"# Seed {seed}")
        print(f"{'#' * 60}")

        seed_dir = Path(args.output_dir) / f"seed_{seed}"

        # Build command
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "train_cv.py"),
            "--seed", str(seed),
            "--output_dir", str(seed_dir),
            "--folds", args.folds,
        ]
        if args.max_epochs is not None:
            cmd += ["--max_epochs", str(args.max_epochs)]
        if args.patience is not None:
            cmd += ["--patience", str(args.patience)]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(project_root))

        if result.returncode != 0:
            print(f"WARNING: Seed {seed} training failed with code {result.returncode}")
            continue

        # Load summary
        summary_path = seed_dir / "cv_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                all_results[f"seed_{seed}"] = json.load(f)

    # Save overall summary
    overall_path = Path(args.output_dir) / "ensemble_summary.json"
    with open(overall_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nEnsemble training complete. Summary saved to {overall_path}")


if __name__ == "__main__":
    main()
