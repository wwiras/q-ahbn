from pathlib import Path
import glob
import pandas as pd


def latest_exp10_csv() -> Path:
    files = sorted(glob.glob("outputs/csv/exp10_results_*.csv"))
    if not files:
        raise FileNotFoundError("No exp10_results_*.csv found.")
    return Path(files[-1])


def main():
    path = latest_exp10_csv()
    print(f"Reading {path}")

    df = pd.read_csv(path)
    q = df[df["strategy"] == "qahbn"].copy()

    cols = [
        "failure_mode",
        "q_table_states",
        "q_pct_recovery_push",
        "q_pct_more_gossip",
        "q_pct_duplicate_suppression",
        "q_pct_resource_conservative",
        "q_pct_ahbn_base",
        "q_pct_more_structured",
    ]

    summary = q.groupby("failure_mode")[cols[1:]].mean().reset_index()

    print("\nQ-AHBN ACTION DIAGNOSTIC")
    print(summary.round(4).to_string(index=False))

    print("\nINTERPRETATION")
    rec = summary["q_pct_recovery_push"].mean()
    gossip = summary["q_pct_more_gossip"].mean()
    suppress = summary["q_pct_duplicate_suppression"].mean()
    conservative = summary["q_pct_resource_conservative"].mean()

    print(f"Average recovery_push             : {rec:.3f}")
    print(f"Average more_gossip               : {gossip:.3f}")
    print(f"Average duplicate_suppression     : {suppress:.3f}")
    print(f"Average resource_conservative     : {conservative:.3f}")

    if rec < 0.15:
        print("\nConclusion: recovery_push is rarely selected.")
        print("Reason recovery did not improve: Q-AHBN is not choosing recovery action often enough.")
        print("Next action: increase recovery reward or strengthen recovery_push.")
    else:
        print("\nConclusion: recovery_push is selected reasonably often.")
        print("Reason recovery did not improve: AHBN recovery is already strong, so Q-learning mainly improves efficiency.")
        print("Next action: stop tuning Exp10-Q and move to Exp11-Q.")


if __name__ == "__main__":
    main()