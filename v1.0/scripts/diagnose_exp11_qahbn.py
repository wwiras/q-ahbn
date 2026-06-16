from pathlib import Path
import glob
import pandas as pd


def latest_exp11_csv() -> Path:
    files = sorted(glob.glob("outputs/csv/exp11_results_*.csv"))
    if not files:
        raise FileNotFoundError("No exp11_results_*.csv found in outputs/csv/")
    return Path(files[-1])


def main() -> None:
    path = latest_exp11_csv()
    print(f"Reading {path}")

    df = pd.read_csv(path)
    q = df[df["strategy"] == "qahbn"].copy()

    cols = [
        "churn_rate",
        "q_table_states",
        "q_pct_recovery_push",
        "q_pct_more_gossip",
        "q_pct_duplicate_suppression",
        "q_pct_resource_conservative",
        "q_pct_ahbn_base",
        "q_pct_more_structured",
    ]

    summary = q.groupby("churn_rate")[cols[1:]].mean().reset_index()

    print("\nQ-AHBN CHURN ACTION DIAGNOSTIC")
    print(summary.round(4).to_string(index=False))

    rec = summary["q_pct_recovery_push"].mean()
    gossip = summary["q_pct_more_gossip"].mean()
    suppress = summary["q_pct_duplicate_suppression"].mean()
    conservative = summary["q_pct_resource_conservative"].mean()

    print("\nINTERPRETATION")
    print(f"Average recovery_push             : {rec:.3f}")
    print(f"Average more_gossip               : {gossip:.3f}")
    print(f"Average duplicate_suppression     : {suppress:.3f}")
    print(f"Average resource_conservative     : {conservative:.3f}")

    if rec + gossip < suppress + conservative:
        print("\nConclusion: Q-AHBN is still too conservative under churn.")
        print("Next action: increase delivery/recovery reward or weaken duplicate penalty.")
    else:
        print("\nConclusion: Q-AHBN is using recovery-oriented actions under churn.")
        print("Next action: compare delay, delivery, and adaptation efficiency against AHBN.")


if __name__ == "__main__":
    main()
