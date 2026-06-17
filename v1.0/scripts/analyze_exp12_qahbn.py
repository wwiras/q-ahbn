from pathlib import Path
import glob
import pandas as pd
import matplotlib.pyplot as plt


def latest_csv(pattern: str) -> Path:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return Path(files[-1])


def latest_exp12_csv() -> Path:
    return latest_csv("outputs/csv/exp12_results_*.csv")


def mean_std(x: pd.Series) -> str:
    return f"{x.mean():.3f} ± {x.std():.3f}"


def add_adaptation_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adaptation Efficiency (AE), higher is better:

        AE = DR / (1 + D_ratio + F_norm + L_norm)

    where:
        DR      = delivery_ratio
        D_ratio = duplicates normalized by max duplicates in the compared set
        F_norm  = total_forwards normalized by max forwards in the compared set
        L_norm  = propagation_delay normalized by max delay in the compared set
    """
    df = df.copy()

    max_dup = max(float(df["duplicates"].max()), 1.0)
    max_fwd = max(float(df["total_forwards"].max()), 1.0)
    max_delay = max(float(df["propagation_delay"].max()), 1.0)

    df["D_ratio"] = df["duplicates"] / max_dup
    df["F_norm"] = df["total_forwards"] / max_fwd
    df["L_norm"] = df["propagation_delay"] / max_delay

    df["adaptation_efficiency"] = (
        df["delivery_ratio"] /
        (1.0 + df["D_ratio"] + df["F_norm"] + df["L_norm"])
    )
    return df


def save_bar(summary: pd.DataFrame, metric: str, ylabel: str, filename: str, timestamp: str) -> None:
    pivot = summary.pivot(index="resource_scenario", columns="strategy", values=metric)

    ax = pivot.plot(kind="bar", figsize=(8, 4.5))
    ax.set_xlabel("Resource heterogeneity scenario")
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


def save_adaptation_efficiency_summary(exp12_df: pd.DataFrame, exp12_timestamp: str) -> None:
    """
    Figure 21: combined adaptation-efficiency summary.
    Uses latest Exp10 and Exp11 result files if available, plus current Exp12.
    """
    frames = []

    def load_phase(pattern: str, phase_name: str) -> None:
        try:
            path = latest_csv(pattern)
        except FileNotFoundError:
            print(f"Skipping {phase_name}: no {pattern} found")
            return
        df = pd.read_csv(path)
        df = df[df["strategy"].isin(["ahbn", "qahbn"])].copy()
        if df.empty:
            return
        df = add_adaptation_efficiency(df)
        tmp = (
            df.groupby("strategy", as_index=False)["adaptation_efficiency"]
            .mean()
        )
        tmp["phase"] = phase_name
        frames.append(tmp)

    load_phase("outputs/csv/exp10_results_*.csv", "Failure")
    load_phase("outputs/csv/exp11_results_*.csv", "Churn")

    df12 = exp12_df[exp12_df["strategy"].isin(["ahbn", "qahbn"])].copy()
    df12 = add_adaptation_efficiency(df12)
    tmp12 = df12.groupby("strategy", as_index=False)["adaptation_efficiency"].mean()
    tmp12["phase"] = "Heterogeneity"
    frames.append(tmp12)

    combined = pd.concat(frames, ignore_index=True)

    out_tables = Path("outputs/tables")
    out_tables.mkdir(parents=True, exist_ok=True)
    table_path = out_tables / f"exp12q_fig21_adaptation_efficiency_summary_{exp12_timestamp}.csv"
    combined.to_csv(table_path, index=False)
    print(f"Saved {table_path}")

    pivot = combined.pivot(index="phase", columns="strategy", values="adaptation_efficiency")
    pivot = pivot.reindex(["Failure", "Churn", "Heterogeneity"])

    ax = pivot.plot(kind="bar", figsize=(7.5, 4.5))
    ax.set_xlabel("Experiment phase")
    ax.set_ylabel("Adaptation Efficiency")
    ax.set_title("Adaptation Efficiency Summary")
    ax.grid(axis="y", alpha=0.3)

    out_fig = Path("outputs/figures")
    out_fig.mkdir(parents=True, exist_ok=True)
    fig_path = out_fig / f"exp12q_fig21_adaptation_efficiency_summary_{exp12_timestamp}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved {fig_path}")


def main() -> None:
    csv_path = latest_exp12_csv()
    print(f"Reading {csv_path}")

    timestamp = csv_path.stem.replace("exp12_results_", "")

    df = pd.read_csv(csv_path)
    df = df[df["strategy"].isin(["ahbn", "qahbn"])].copy()
    df = add_adaptation_efficiency(df)

    summary = (
        df.groupby(["resource_scenario", "strategy"])
        .agg(
            delivery_ratio=("delivery_ratio", "mean"),
            propagation_delay=("propagation_delay", "mean"),
            duplicates=("duplicates", "mean"),
            total_forwards=("total_forwards", "mean"),
            max_normalized_load=("max_normalized_load", "mean"),
            load_balance_cv=("load_balance_cv", "mean"),
            strong_forward_share=("strong_forward_share", "mean"),
            medium_forward_share=("medium_forward_share", "mean"),
            weak_forward_share=("weak_forward_share", "mean"),
            adaptation_efficiency=("adaptation_efficiency", "mean"),
            q_table_states=("q_table_states", "mean"),
            q_updates=("q_updates", "mean"),
            q_mean_reward=("q_mean_reward", "mean"),
            q_recent_reward=("q_recent_reward", "mean"),
            q_pct_resource_conservative=("q_pct_resource_conservative", "mean"),
            q_pct_duplicate_suppression=("q_pct_duplicate_suppression", "mean"),
            q_pct_more_structured=("q_pct_more_structured", "mean"),
            q_pct_more_gossip=("q_pct_more_gossip", "mean"),
        )
        .reset_index()
    )

    out = Path("outputs/tables")
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / f"exp12q_summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    paper_table = (
        df.groupby(["resource_scenario", "strategy"])
        .agg({
            "delivery_ratio": mean_std,
            "propagation_delay": mean_std,
            "duplicates": mean_std,
            "total_forwards": mean_std,
            "max_normalized_load": mean_std,
            "load_balance_cv": mean_std,
            "strong_forward_share": mean_std,
            "medium_forward_share": mean_std,
            "weak_forward_share": mean_std,
            "adaptation_efficiency": mean_std,
        })
        .reset_index()
    )

    paper_table_path = out / f"exp12q_paper_table_{timestamp}.csv"
    paper_table.to_csv(paper_table_path, index=False)
    print(f"Saved {paper_table_path}")

    # Paper mapping requested by user.
    save_bar(summary, "propagation_delay", "Normal vs low-resource delay", "exp12q_fig15_normal_vs_low_resource_delay.png", timestamp)
    save_bar(summary, "duplicates", "Duplicates", "exp12q_fig16_duplicates.png", timestamp)
    save_bar(summary, "total_forwards", "Forwarding cost", "exp12q_fig17_forwarding_cost.png", timestamp)
    save_bar(summary, "adaptation_efficiency", "Adaptation Efficiency", "exp12q_fig21_adaptation_efficiency_heterogeneity.png", timestamp)

    # Combined Figure 21: Failure + Churn + Heterogeneity.
    save_adaptation_efficiency_summary(df, timestamp)

    print("\nEXP12-Q SUMMARY")
    print(summary.round(4).to_string(index=False))

    print("\nPAPER MAPPING")
    print(f"Table 12 : {paper_table_path}")
    print("Figure 15: exp12q_fig15_normal_vs_low_resource_delay_<timestamp>.png")
    print("Figure 16: exp12q_fig16_duplicates_<timestamp>.png")
    print("Figure 17: exp12q_fig17_forwarding_cost_<timestamp>.png")
    print("Figure 21a: exp12q_fig21_adaptation_efficiency_heterogeneity_<timestamp>.png")
    print("Figure 21 : exp12q_fig21_adaptation_efficiency_summary_<timestamp>.png")


if __name__ == "__main__":
    main()
