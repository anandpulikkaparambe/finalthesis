"""Turn the raw per-step CSVs (ur5e_grasp_env.py's hardware_log_env*.csv, evaluate.py's
eval_episodes.csv) into report-ready figures. No Isaac Sim needed -- pure CSV/matplotlib,
run with the regular system Python, not python.bat.

    python scripts/plot_training.py --log-dir rl_logs/run1_2026-09-22 --out-dir report_figs
    python scripts/plot_training.py --eval-csv eval_out/eval_episodes.csv --out-dir report_figs
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--log-dir", type=str, default="", help="Dir with hardware_log_env*.csv (training)")
parser.add_argument("--eval-csv", type=str, default="", help="evaluate.py's eval_episodes.csv")
parser.add_argument("--out-dir", type=str, default="./report_figs")
parser.add_argument("--smooth", type=int, default=200, help="rolling-mean window (steps) for the noisy per-step curves")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)


def rolling_mean(xs, window):
    out, s, q = [], 0.0, []
    for x in xs:
        q.append(x)
        s += x
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def read_training_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("Global_Step", "Curriculum_Lvl", "Dist_to_Target", "Reward", "Self_Clearance",
                   "Arm_Table_Clearance", "Contact_L_N", "Contact_R_N", "Spawn_Radius_M", "Rand_Strength"):
            r[k] = float(r[k])
        r["Terminated"] = r["Terminated"] == "True"
    return rows


def plot_training(log_dir, out_dir, smooth):
    paths = sorted(glob.glob(os.path.join(log_dir, "hardware_log_env*.csv")))
    if not paths:
        print(f"no hardware_log_env*.csv found under {log_dir}", flush=True)
        return
    for path in paths:
        rows = read_training_csv(path)
        if not rows:
            continue
        tag = os.path.splitext(os.path.basename(path))[0]
        steps = [r["Global_Step"] for r in rows]

        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        axes[0].plot(steps, rolling_mean([r["Reward"] for r in rows], smooth), color="tab:blue")
        axes[0].set_ylabel(f"Reward ({smooth}-step mean)")
        axes[0].set_title(f"Training progress -- {tag}")
        axes[0].grid(alpha=0.3)

        axes[1].plot(steps, rolling_mean([r["Dist_to_Target"] for r in rows], smooth), color="tab:orange")
        axes[1].set_ylabel("Dist to target (m)")
        axes[1].grid(alpha=0.3)

        axes[2].plot(steps, [r["Curriculum_Lvl"] for r in rows], color="tab:green", label="Curriculum level")
        axes[2].plot(steps, [r["Spawn_Radius_M"] for r in rows], color="tab:red", alpha=0.6, label="Spawn radius (m)")
        axes[2].set_ylabel("Curriculum / spawn radius")
        axes[2].set_xlabel("Global step")
        axes[2].legend(loc="upper left", fontsize=8)
        axes[2].grid(alpha=0.3)

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{tag}_progress.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}", flush=True)

        # Termination-reason histogram over episodes (each Terminated=True row is one episode end)
        term_steps = [r["Global_Step"] for r in rows if r["Terminated"]]
        term_dist = [r["Dist_to_Target"] for r in rows if r["Terminated"]]
        if term_steps:
            fig2, ax = plt.subplots(figsize=(9, 4))
            ax.scatter(term_steps, term_dist, s=10, alpha=0.6, color="tab:purple")
            ax.set_xlabel("Global step")
            ax.set_ylabel("Distance to target at termination (m)")
            ax.set_title(f"Episode-end distance over training -- {tag}")
            ax.grid(alpha=0.3)
            fig2.tight_layout()
            out_path2 = os.path.join(out_dir, f"{tag}_termination_distance.png")
            fig2.savefig(out_path2, dpi=150)
            plt.close(fig2)
            print(f"wrote {out_path2}", flush=True)

        # Contact force trace (evidence of grasp attempts, even before any confirmed success)
        fig3, ax = plt.subplots(figsize=(9, 4))
        ax.plot(steps, [r["Contact_L_N"] for r in rows], label="Left pad", alpha=0.7, linewidth=0.6)
        ax.plot(steps, [r["Contact_R_N"] for r in rows], label="Right pad", alpha=0.7, linewidth=0.6)
        ax.set_xlabel("Global step")
        ax.set_ylabel("Contact force (N)")
        ax.set_title(f"Finger-pad contact force -- {tag}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig3.tight_layout()
        out_path3 = os.path.join(out_dir, f"{tag}_contact_force.png")
        fig3.savefig(out_path3, dpi=150)
        plt.close(fig3)
        print(f"wrote {out_path3}", flush=True)


def plot_eval(eval_csv, out_dir):
    with open(eval_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{eval_csv} is empty", flush=True)
        return
    successes = [r["episode"] for r in rows if r["success"] == "True"]
    reasons = {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(reasons.keys(), reasons.values(), color="tab:blue")
    ax.set_ylabel("Episode count")
    ax.set_title(f"Evaluation termination reasons (n={len(rows)}, {len(successes)} successes)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "eval_termination_reasons.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)

    dists = sorted(float(r["min_distance"]) for r in rows)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(range(len(dists)), dists, marker="o", markersize=3)
    ax2.set_xlabel("Episode (sorted)")
    ax2.set_ylabel("Min distance to target (m)")
    ax2.set_title("Per-episode closest approach, sorted")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    out_path2 = os.path.join(out_dir, "eval_min_distance_sorted.png")
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)
    print(f"wrote {out_path2}", flush=True)


if args.log_dir:
    plot_training(args.log_dir, args.out_dir, args.smooth)
if args.eval_csv:
    plot_eval(args.eval_csv, args.out_dir)
if not args.log_dir and not args.eval_csv:
    parser.error("pass --log-dir and/or --eval-csv")
