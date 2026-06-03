from __future__ import annotations

import sys
import pandas as pd


def main(path: str) -> None:
    df = pd.read_csv(path)
    group_cols = [c for c in ["experiment", "strategy", "fanout", "edge_prob", "ch_overload_factor"] if c in df.columns]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            delivery_ratio_mean=("delivery_ratio", "mean"),
            propagation_delay_mean=("propagation_delay", "mean"),
            duplicates_mean=("duplicates", "mean"),
            total_forwards_mean=("total_forwards", "mean"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/summarize_results.py outputs/csv/exp07_results.csv")
        raise SystemExit(1)
    main(sys.argv[1])