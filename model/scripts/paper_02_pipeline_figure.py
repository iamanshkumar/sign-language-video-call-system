"""Paper figure F1: system pipeline (offline training/evaluation and online inference).

  PYTHONPATH=src .venv/bin/python scripts/paper_02_pipeline_figure.py --out reports/paper/draft

Writes F1_pipeline.{svg,pdf,png}. Line style encodes status, taken from the repository state:
solid = implemented and used for the reported results; dashed = implemented, evaluated offline only (not in
the live pipeline / defaults off); dotted grey = proposed or not implemented. Refuses to overwrite.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from asl.config import PROJECT_ROOT  # noqa: E402

STYLE = {"impl": dict(ls="-", ec="black", fc="white", lw=1.2, tc="black"),
         "offline": dict(ls="--", ec="black", fc="white", lw=1.2, tc="black"),
         "proposed": dict(ls=":", ec="0.55", fc="0.96", lw=1.2, tc="0.4")}


def box(ax, x, y, w, h, title, body, status):
    s = STYLE[status]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linestyle=s["ls"], edgecolor=s["ec"], facecolor=s["fc"], linewidth=s["lw"]))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=7.6, fontweight="bold", color=s["tc"])
    ax.text(x + w / 2, y + h - 0.40, body, ha="center", va="top", fontsize=6.3, color=s["tc"], linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, status="impl"):
    s = STYLE[status]
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, linewidth=1.0,
                                 linestyle=s["ls"], color=s["ec"]))


def draw(out: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.set_xlim(0, 23); ax.set_ylim(0, 13); ax.axis("off")
    W, H = 3.4, 2.15
    # lane labels
    ax.text(0.15, 12.6, "A. Offline: data, training and evaluation", fontsize=9, fontweight="bold")
    ax.text(0.15, 5.95, "B. Online: streaming inference (one signer)", fontsize=9, fontweight="bold")
    ax.plot([0.1, 22.9], [6.35, 6.35], color="0.8", lw=0.8)
    yA1, yA2 = 9.9, 6.9
    # --- offline lane, row 1
    box(ax, 0.2, yA1, W, H, "Dataset", "Hugging Face ASL videos\n100 word classes, 4,838 clips\nfrozen split 3,407 / 725 / 706\n(not signer-independent)", "impl")
    box(ax, 4.1, yA1, W, H, "Landmark extraction", "MediaPipe Holistic\npose (33) + both hands (21 each)\nface NOT extracted\n(hands+pose+face not evaluated)", "impl")
    box(ax, 8.0, yA1, W, H, "Preprocessing", "30 fps resampling, rejection rules\nactive signing interval\nshoulder-centred, shoulder-\nwidth-scaled coordinates", "impl")
    box(ax, 11.9, yA1, W, H, "Temporal sequences", "active interval resampled to\nW = 30 frames (W = 45: one\nLSTM seed, exploratory)\nno sliding window", "impl")
    box(ax, 15.8, yA1, W, H, "Feature sets", "hands + pose: 261\nhands only: 128\nhands + upper-body pose\n(landmarks 0-22): 221", "impl")
    box(ax, 19.5, yA1, 3.3, H, "Training", "LSTM / BiLSTM, 2 x 128\nAdamW, early stopping\ncheckpoint on validation\nmacro-F1, 5 seeds", "impl")
    for x in (3.6, 7.5, 11.4, 15.3):
        arrow(ax, x, yA1 + H / 2, x + 0.5, yA1 + H / 2)
    arrow(ax, 19.2, yA1 + H / 2, 19.5, yA1 + H / 2)
    # --- offline lane, row 2
    box(ax, 19.5, yA2, 3.3, 2.0, "Augmentation (E1)", "training split only, p = 0.5:\nhold start/end, tempo warp,\nloop-closed repetition", "impl")
    box(ax, 15.8, yA2, W, 2.0, "Upper-body ablation (E2)", "pose landmarks 23-32 removed\n(261 -> 221 features)\nnon-inferiority rule (validation)", "impl")
    box(ax, 11.9, yA2, W, 2.0, "Evaluation", "validation: all decisions\ntest: once, after development\n(earlier test uses disclosed)\nbootstrap CIs, paired seeds", "impl")
    box(ax, 8.0, yA2, W, 2.0, "Export", "baseline BiLSTM (261) exported\nas live candidate; E1/E2 not\nexported (E2: 221 features)", "impl")
    box(ax, 4.1, yA2, W, 2.0, "Robustness diagnostics", "validation perturbations:\nholds, repetition, hand loss,\nmirror, rotation, hips", "impl")
    arrow(ax, 21.15, yA1, 21.15, yA2 + 2.0)
    arrow(ax, 19.5, yA2 + 1.0, 19.2, yA2 + 1.0)
    arrow(ax, 15.8, yA2 + 1.0, 15.3, yA2 + 1.0)
    arrow(ax, 11.9, yA2 + 1.0, 11.4, yA2 + 1.0)
    ax.add_patch(FancyArrowPatch((12.6, yA2 + 2.0), (5.8, yA2 + 2.0), arrowstyle="-|>", mutation_scale=8, linewidth=1.0,
                                 color="black", connectionstyle="arc3,rad=0.25"))
    # --- online lane
    yB = 2.75
    box(ax, 0.2, yB, 2.9, 2.6, "Webcam capture", "OpenCV, 640x480 requested\nthreaded capture,\nnewest frame\n(15.0 -> 29.8 fps)", "impl")
    box(ax, 3.5, yB, 2.9, 2.6, "MediaPipe Holistic", "per frame\n28.5 ms median\n(CPU laptop)", "impl")
    box(ax, 6.8, yB, 3.1, 2.6, "Segmentation", "hand-activity segments:\npre-roll 0.5 s, end after\n0.4 s without hands,\ncap 4.0 s", "impl")
    box(ax, 10.3, yB, 3.1, 2.6, "Classifier", "training preprocessing,\nW = 30; baseline BiLSTM\n(261 features)\n1.68 ms per sign", "impl")
    box(ax, 13.8, yB, 3.2, 2.6, "Post-processing", "cooldown, confidence\nthreshold: replay-evaluated,\nOFF in live defaults;\nsmoothing not evaluated", "offline")
    box(ax, 17.4, yB, 2.4, 2.6, "Word label", "emitted ~0.44-0.46 s\nafter hands are\nlowered", "impl")
    box(ax, 20.2, yB, 2.6, 2.6, "Video call", "Next.js + WebRTC\ncalls exist; FastAPI\ninference service and\ndelivery: NOT built", "proposed")
    for x1, x2 in ((3.1, 3.5), (6.4, 6.8), (9.9, 10.3), (13.4, 13.8), (17.0, 17.4)):
        arrow(ax, x1, yB + 1.3, x2, yB + 1.3)
    arrow(ax, 19.8, yB + 1.3, 20.2, yB + 1.3, "proposed")
    box(ax, 10.3, 0.15, 6.7, 1.95, "E1 / E2 models in the live pipeline", "not integrated: inference accepts 261-feature checkpoints only;\nE2 webcam results come from offline replay of saved landmarks", "proposed")
    arrow(ax, 9.7, yA2, 11.85, yB + 2.6)
    arrow(ax, 13.65, 2.1, 12.4, yB, "proposed")
    # legend
    lx, ly = 0.2, 0.25
    for i, (k, lab) in enumerate((("impl", "implemented, used for reported results"),
                                  ("offline", "implemented, evaluated offline only / off by default"),
                                  ("proposed", "proposed or not integrated"))):
        s = STYLE[k]
        ax.add_patch(FancyBboxPatch((lx, ly + i * 0.62), 0.7, 0.4, boxstyle="round,pad=0.02", linestyle=s["ls"],
                                    edgecolor=s["ec"], facecolor=s["fc"], linewidth=s["lw"]))
        ax.text(lx + 0.95, ly + i * 0.62 + 0.2, lab, fontsize=6.8, va="center", color=s["tc"])
    fig.tight_layout()
    paths = [out / f"F1_pipeline.{ext}" for ext in ("svg", "pdf", "png")]
    for p in paths:
        fig.savefig(p, dpi=200 if p.suffix == ".png" else None)
    plt.close(fig)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    out = PROJECT_ROOT / ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    if any((out / f"F1_pipeline.{e}").exists() for e in ("svg", "pdf", "png")):
        raise SystemExit("F1_pipeline already exists in --out; choose a new folder")
    for p in draw(out):
        print(p.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
