from pathlib import Path
import glob
import pandas as pd
import matplotlib.pyplot as plt


def latest_exp11_csv() -> Path:
    files = sorted(glob.glob("outputs/csv/exp11_results_*.csv"))
    if not files:
        raise FileNotFoundError("No exp11_results_*.csv found in outputs/csv/")
    return Path(files[-1])


def mean_std(x: pd.Series) -> str:
    return f"{x.mean():.3f} ± {x.std():.3f}"


def add_adaptation_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    max_dup = max(float(df["duplicates"].max()), 1.0)
    max_fwd = max(float(df["total_forwards"].max()), 1.0)
    max_delay = max(float(df["propagation_delay"].max()), 1.0)

    df["D_ratio"] = df["duplicates"] / max_dup
    df["F_norm"] = df["total_forwards"] / max_fwd
    df["L_norm"] = df["propagation_delay"] / max_delay

    # Higher is better.
    # AE = DR / (1 + normalized duplication + normalized forwarding + normalized latency)
    df["adaptation_efficiency"] = (
        df["delivery_ratio"] /
        (1.0 + df["D_ratio"] + df["F_norm"] + df["L_norm"])
    )

    return df


def save_line(summary: pd.DataFrame, metric: str, ylabel: str, filename: str, timestamp: str) -> None:
    pivot = summary.pivot(index="churn_rate", columns="strategy", values=metric)

    ax = pivot.plot(kind="line", marker="o", figsize=(7, 4))
    ax.set_xlabel("Churn rate")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(axis="y", alpha=0.3)

    out = Path("outputs/figures")
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename.replace(".png", f"_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def save_bar(summary: pd.DataFrame, metric: str, ylabel: str, filename: str, timestamp: str) -> None:
    pivot = summary.pivot(index="churn_rate", columns="strategy", values=metric)

    ax = pivot.plot(kind="bar", figsize=(7, 4))
    ax.set_xlabel("Churn rate")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(axis="y", alpha=0.3)

    out = Path("outputs/figures")
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename.replace(".png", f"_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def main() -> None:
    csv_path = latest_exp11_csv()
    print(f"Reading {csv_path}")

    timestamp = csv_path.stem.replace("exp11_results_", "")

    df = pd.read_csv(csv_path)
    df = df[df["strategy"].isin(["ahbn", "qahbn"])].copy()
    df = add_adaptation_efficiency(df)

    summary = (
        df.groupby(["churn_rate", "strategy"])
        .agg(
            delivery_ratio=("delivery_ratio", "mean"),
            propagation_delay=("propagation_delay", "mean"),
            churn_recovery_time=("churn_recovery_time", "mean"),
            duplicates=("duplicates", "mean"),
            total_forwards=("total_forwards", "mean"),
            adaptation_rate=("adaptation_rate", "mean"),
            adaptation_efficiency=("adaptation_efficiency", "mean"),
            q_table_states=("q_table_states", "mean"),
            q_updates=("q_updates", "mean"),
            q_mean_reward=("q_mean_reward", "mean"),
            q_recent_reward=("q_recent_reward", "mean"),
            q_pct_recovery_push=("q_pct_recovery_push", "mean"),
            q_pct_more_gossip=("q_pct_more_gossip", "mean"),
            q_pct_duplicate_suppression=("q_pct_duplicate_suppression", "mean"),
        )
        .reset_index()
    )

    out = Path("outputs/tables")
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / f"exp11q_summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    paper_table = (
        df.groupby(["churn_rate", "strategy"])
        .agg({
            "delivery_ratio": mean_std,
            "propagation_delay": mean_std,
            "churn_recovery_time": mean_std,
            "duplicates": mean_std,
            "total_forwards": mean_std,
            "adaptation_rate": mean_std,
            "adaptation_efficiency": mean_std,
        })
        .reset_index()
    )

    paper_table_path = out / f"exp11q_paper_table_{timestamp}.csv"
    paper_table.to_csv(paper_table_path, index=False)
    print(f"Saved {paper_table_path}")

    # Paper mapping requested by user.
    save_line(summary, "propagation_delay", "Delay under churn", "exp11q_fig12_delay_under_churn.png", timestamp)
    save_line(summary, "churn_recovery_time", "Recovery under churn", "exp11q_fig13_recovery_under_churn.png", timestamp)
    save_line(summary, "delivery_ratio", "Delivery ratio", "exp11q_fig14_delivery_ratio.png", timestamp)
    save_bar(summary, "adaptation_efficiency", "Adaptation Efficiency", "exp11q_fig21_adaptation_efficiency_churn.png", timestamp)

    print("\nEXP11-Q SUMMARY")
    print(summary.round(4).to_string(index=False))

    print("\nPAPER MAPPING")
    print(f"Table 11 : {paper_table_path}")
    print("Figure 12: exp11q_fig12_delay_under_churn_<timestamp>.png")
    print("Figure 13: exp11q_fig13_recovery_under_churn_<timestamp>.png")
    print("Figure 14: exp11q_fig14_delivery_ratio_<timestamp>.png")
    print("Figure 21: exp11q_fig21_adaptation_efficiency_churn_<timestamp>.png")


if __name__ == "__main__":
    main()
