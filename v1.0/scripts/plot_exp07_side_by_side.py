from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def extract_timestamp_from_filename(path: Path) -> str:
    """
    Extract trailing timestamp from filenames such as:
    exp07_results_20260407_140852.csv
    If not found, return 'latest'.
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 4:
        maybe_date = parts[-2]
        maybe_time = parts[-1]
        if maybe_date.isdigit() and maybe_time.isdigit():
            return f"{maybe_date}_{maybe_time}"
    return "latest"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to exp07 results CSV")
    parser.add_argument(
        "--output-dir",
        default="outputs/plots",
        help="Directory to save the generated figure",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required_cols = {
        "experiment",
        "strategy",
        "fanout",
        "propagation_delay",
        "duplicates",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    # Keep only Exp07 and only the two strategies we want
    df = df[df["experiment"] == "exp07"].copy()
    df = df[df["strategy"].isin(["gossip", "ahbn"])].copy()

    if df.empty:
        raise ValueError("No Exp07 rows found for strategies 'gossip' and 'ahbn'.")

    # Aggregate mean and std across runs for each strategy/fanout
    grouped = (
        df.groupby(["strategy", "fanout"], as_index=False)
        .agg(
            delay_mean=("propagation_delay", "mean"),
            delay_std=("propagation_delay", "std"),
            dup_mean=("duplicates", "mean"),
            dup_std=("duplicates", "std"),
        )
        .sort_values(["strategy", "fanout"])
    )

    gossip = grouped[grouped["strategy"] == "gossip"].sort_values("fanout")
    ahbn = grouped[grouped["strategy"] == "ahbn"].sort_values("fanout")

    if gossip.empty or ahbn.empty:
        raise ValueError("Both 'gossip' and 'ahbn' must be present in the CSV.")

    timestamp = extract_timestamp_from_filename(csv_path)
    output_path = output_dir / f"exp07_gossip_vs_ahbn_side_by_side_{timestamp}.png"

    plt.figure(figsize=(12, 5))

    # -----------------------------
    # Left subplot: Delay
    # -----------------------------
    plt.subplot(1, 2, 1)
    plt.plot(
        gossip["fanout"],
        gossip["delay_mean"],
        marker="o",
        label="Gossip",
    )
    plt.plot(
        ahbn["fanout"],
        ahbn["delay_mean"],
        marker="o",
        label="AHBN",
    )

    # Optional error bars
    if gossip["delay_std"].notna().any():
        plt.errorbar(
            gossip["fanout"],
            gossip["delay_mean"],
            yerr=gossip["delay_std"].fillna(0),
            fmt="none",
            capsize=3,
        )
    if ahbn["delay_std"].notna().any():
        plt.errorbar(
            ahbn["fanout"],
            ahbn["delay_mean"],
            yerr=ahbn["delay_std"].fillna(0),
            fmt="none",
            capsize=3,
        )

    plt.title("Exp07: Delay vs Fanout")
    plt.xlabel("Fanout")
    plt.ylabel("Average Propagation Delay")
    plt.xticks(sorted(df["fanout"].dropna().unique()))
    plt.grid(True, alpha=0.3)
    plt.legend()

    # -----------------------------
    # Right subplot: Duplicates
    # -----------------------------
    plt.subplot(1, 2, 2)
    plt.plot(
        gossip["fanout"],
        gossip["dup_mean"],
        marker="o",
        label="Gossip",
    )
    plt.plot(
        ahbn["fanout"],
        ahbn["dup_mean"],
        marker="o",
        label="AHBN",
    )

    # Optional error bars
    if gossip["dup_std"].notna().any():
        plt.errorbar(
            gossip["fanout"],
            gossip["dup_mean"],
            yerr=gossip["dup_std"].fillna(0),
            fmt="none",
            capsize=3,
        )
    if ahbn["dup_std"].notna().any():
        plt.errorbar(
            ahbn["fanout"],
            ahbn["dup_mean"],
            yerr=ahbn["dup_std"].fillna(0),
            fmt="none",
            capsize=3,
        )

    plt.title("Exp07: Duplicates vs Fanout")
    plt.xlabel("Fanout")
    plt.ylabel("Average Duplicates")
    plt.xticks(sorted(df["fanout"].dropna().unique()))
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.suptitle("Experiment 07: Gossip vs AHBN", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()