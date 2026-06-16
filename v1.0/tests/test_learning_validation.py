# tests/test_learning_validation.py

from collections import Counter
import csv
from pathlib import Path
from datetime import datetime
import random

from ahbn.control import (
    AHBNController,
    AHBNParams,
    NodeControlState,
)

from ahbn.q_learning import QAHBNController


# EPISODES = 200
EPISODES = 500


# def create_state(ep: int) -> NodeControlState:
#     """
#     Simulated environment.

#     Early episodes:
#         poor network conditions

#     Later episodes:
#         mixed conditions

#     Allows Q-learning to discover
#     better actions over time.
#     """

#     if ep < 50:
#         return NodeControlState(
#             d_hat=0.60,
#             u_hat=4.0,
#             l_hat=3.0,
#             rho_hat=0.40,
#             r_hat=0.60,
#             c_hat=0.50,
#             delivery_estimate=0.20,
#         )

#     elif ep < 100:
#         return NodeControlState(
#             d_hat=0.40,
#             u_hat=2.5,
#             l_hat=2.0,
#             rho_hat=0.25,
#             r_hat=0.40,
#             c_hat=0.30,
#             delivery_estimate=0.50,
#         )

#     else:
#         return NodeControlState(
#             d_hat=0.10,
#             u_hat=1.0,
#             l_hat=1.0,
#             rho_hat=0.05,
#             r_hat=0.10,
#             c_hat=0.10,
#             delivery_estimate=0.90,
#         )


def create_state(ep: int) -> NodeControlState:
    return NodeControlState(
        d_hat=random.uniform(0.0, 1.0),
        u_hat=random.uniform(0.5, 8.0),
        l_hat=random.uniform(0.5, 5.0),
        rho_hat=random.uniform(0.0, 1.0),
        r_hat=random.uniform(0.0, 1.0),
        c_hat=random.uniform(0.0, 1.0),
        delivery_estimate=random.uniform(0.2, 1.0),
)

def main():

    params = AHBNParams()

    base = AHBNController(params)

    q = QAHBNController(
        base_controller=base,
        cfg={},
        seed=42,
    )

    action_counter = Counter()

    # csv_file = "learning_trace.csv"
    Path("outputs/csv").mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_file = (
        f"outputs/csv/"
        f"learning_trace_{timestamp}.csv"
    )

    with open(csv_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "episode",
            "reward",
            "epsilon",
            "action",
            "state",
        ])

        for ep in range(EPISODES):

            state = create_state(ep)

            q.decide_mode_and_fanout(state)

            reward = q.compute_reward(state)

            action = getattr(state, "q_action", "unknown")

            action_counter[action] += 1

            writer.writerow([
                ep + 1,
                reward,
                q.epsilon,
                action,
                getattr(state, "q_state", ""),
            ])

    summary = q.get_learning_summary()

    print("\n==============================")
    print("Q-AHBN LEARNING VALIDATION")
    print("==============================")

    print(f"Episodes            : {EPISODES}")
    print(f"Q Updates           : {summary['q_updates']}")
    print(f"States Learned      : {summary['q_table_states']}")
    print(f"Mean Reward         : {summary['q_mean_reward']:.3f}")
    print(f"Recent Reward       : {summary['q_recent_reward']:.3f}")
    print(f"Final Epsilon       : {summary['q_epsilon_final']:.3f}")

    print("\nACTION DISTRIBUTION")

    for action, count in sorted(
        action_counter.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"{action:25s} {count:4d}")

    print(f"\nCSV Written: {csv_file}")


if __name__ == "__main__":
    main()