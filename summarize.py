"""Collect the results of all runs into one table and one plot.

  python summarize.py   # writes runs/summary.md and runs/fid_curves.png
"""

import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"


def load(path):
    return json.loads(path.read_text()) if path.exists() else []


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, curves = [], {}
    for run in sorted(p for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith("_")):
        args = load(run / "args.json") or {}
        bpe = (load(run / "history.json") or {}).get("batches_per_epoch", 637)
        for r in load(run / "metrics.json"):
            rows.append((run.name, r, args))
        for r in load(run / "metrics_all.json"):
            it = r.get("iteration") or r["epoch"] * bpe  # older runs have no iteration
            curves.setdefault((run.name, r.get("G", "G")), []).append((it, r["fid"]))
        for f in sorted(run.glob("metrics_*.json")):
            if f.name != "metrics_all.json":
                for r in load(f):
                    label = f"vs {r['style']}" if r.get("style") else f"@ epoch {r['epoch']}"
                    rows.append((f"{run.name} {label}", r, args))

    lines = ["| run | G | epoch | styles | n | FID | FID (ignite) | IS |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r, args in rows:
        styles = r.get("style") or ", ".join(args.get("styles") or ["all"])
        lines.append(f"| {name} | {r.get('G', 'G')} | {r['epoch']} | {styles} | {r['n']} | {r['fid']:.1f} | "
                     f"{r['fid_ignite']:.4f} | {r['is']:.2f} |")
    real_is = [r["is_real"] for _, r, a in rows if "is_real" in r and not a.get("styles")]
    if real_is:
        lines.append(f"\nIS of real images (all styles): {real_is[0]:.2f}")
    (RUNS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    plt.figure(figsize=(11, 6))
    for (name, g), pts in sorted(curves.items()):
        pts.sort()
        plt.plot([p[0] for p in pts], [p[1] for p in pts], label=name + (" (EMA)" if g == "G_ema" else ""),
                 linestyle="--" if g == "G_ema" or name.startswith("style_") else "-")
    plt.xlabel("iterations")
    plt.ylabel("FID (2k images, lower is better)")
    plt.ylim(top=200)
    plt.legend(fontsize=8, ncol=2)
    plt.title("FID during training")
    plt.savefig(RUNS / "fid_curves.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
