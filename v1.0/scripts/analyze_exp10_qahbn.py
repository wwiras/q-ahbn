from pathlib import Path
import glob
import pandas as pd
import matplotlib.pyplot as plt


def latest_exp10_csv() -> Path:
    files = sorted(glob.glob("outputs/csv/exp10_results_*.csv"))
    if not files:
        raise FileNotFoundError("No exp10_results_*.csv found in outputs/csv/")
    return Path(files[-1])


def mean_std(x):
    return f"{x.mean():.3f} ± {x.std():.3f}"


def add_adaptation_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    max_dup = max(df["duplicates"].max(), 1)
    max_fwd = max(df["total_forwards"].max(), 1)
    max_delay = max(df["propagation_delay"].max(), 1)

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


# def save_bar(summary, metric, ylabel, filename):
def save_bar(summary, metric, ylabel, filename, timestamp):
    pivot = summary.pivot(index="failure_mode", columns="strategy", values=metric)

    ax = pivot.plot(kind="bar", figsize=(7, 4))
    ax.set_xlabel("Failure scenario")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = Path("outputs/figures")
    out.mkdir(parents=True, exist_ok=True)
    name = filename.replace(
        ".png",
        f"_{timestamp}.png"
    )

    # path = out / filename
    path = out / name
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def main():
    # csv_path = latest_exp10_csv()
    # print(f"Reading {csv_path}")
    csv_path = latest_exp10_csv()
    print(f"Reading {csv_path}")

    timestamp = csv_path.stem.replace("exp10_results_", "")

    df = pd.read_csv(csv_path)
    df = df[df["strategy"].isin(["ahbn", "qahbn"])].copy()
    df = add_adaptation_efficiency(df)

    summary = (
        df.groupby(["failure_mode", "strategy"])
        .agg(
            delivery_ratio=("delivery_ratio", "mean"),
            propagation_delay=("propagation_delay", "mean"),
            recovery_time=("recovery_time", "mean"),
            duplicates=("duplicates", "mean"),
            total_forwards=("total_forwards", "mean"),
            adaptation_efficiency=("adaptation_efficiency", "mean"),
            q_table_states=("q_table_states", "mean"),
            q_updates=("q_updates", "mean"),
            q_mean_reward=("q_mean_reward", "mean"),
            q_recent_reward=("q_recent_reward", "mean"),
        )
        .reset_index()
    )

    out = Path("outputs/tables")
    out.mkdir(parents=True, exist_ok=True)

    # summary_path = out / "table10_exp10_failure_summary.csv"
    # summary_path = (
    #     out /
    #     f"table10_exp10_failure_summary_{timestamp}.csv"
    # )
    summary_path = (
        out /
        f"exp10q_summary_{timestamp}.csv"
    )
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    paper_table = (
        df.groupby(["failure_mode", "strategy"])
        .agg({
            "delivery_ratio": mean_std,
            "propagation_delay": mean_std,
            "recovery_time": mean_std,
            "duplicates": mean_std,
            "total_forwards": mean_std,
            "adaptation_efficiency": mean_std,
        })
        .reset_index()
    )

    # paper_table_path = out / "table10_exp10_failure_paper_ready.csv"
    # paper_table_path = (
    #     out /
    #     f"table10_exp10_failure_paper_ready_{timestamp}.csv"
    # )
    paper_table_path = (
        out /
        f"exp10q_paper_table_{timestamp}.csv"
    )
    paper_table.to_csv(paper_table_path, index=False)
    print(f"Saved {paper_table_path}")

    # save_bar(summary, "recovery_time", "Recovery Time", "fig09_recovery_time_comparison.png",timestamp)
    # save_bar(summary, "duplicates", "Duplicates", "fig10_duplicates_comparison.png",timestamp)
    # save_bar(summary, "delivery_ratio", "Delivery Ratio", "fig11_delivery_comparison.png",timestamp)
    # save_bar(summary, "adaptation_efficiency", "Adaptation Efficiency", "fig21_adaptation_efficiency_failure.png",timestamp)
    
    save_bar(summary, "recovery_time", "Recovery Time", "exp10q_recovery_time.png",timestamp)
    save_bar(summary, "duplicates", "Duplicates", "exp10q_duplicates.png",timestamp)
    save_bar(summary, "delivery_ratio", "Delivery Ratio", "exp10q_delivery_ratio.png",timestamp)
    save_bar(summary, "adaptation_efficiency", "Adaptation Efficiency", "exp10q_adaptation_efficiency.png",timestamp)

    print("\nEXP10-Q SUMMARY")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()