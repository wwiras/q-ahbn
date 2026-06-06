from ahbn.control import AHBNController, AHBNParams, NodeControlState
from ahbn.q_learning import QAHBNController


def main():
    params = AHBNParams()
    base = AHBNController(params)

    q = QAHBNController(
        base_controller=base,
        cfg={
            "alpha": 0.25,
            "gamma": 0.90,
            "epsilon": 0.0,
            "epsilon_min": 0.0,
            "epsilon_decay": 1.0,
        },
        seed=42,
    )

    state = NodeControlState(
        d_hat=0.20,
        u_hat=1.00,
        l_hat=1.00,
        rho_hat=0.05,
        r_hat=0.10,
        c_hat=0.10,
        delivery_estimate=0.50,
    )

    print("\n==============================")
    print("Q-TABLE UPDATE TEST")
    print("==============================")

    # First decision: creates previous state-action pair, but no Q-update yet.
    q.decide_mode_and_fanout(state)

    prev_state, prev_action = q.prev[id(state)]
    q_before = q.q_table[prev_state][prev_action]

    print(f"Previous state  : {prev_state}")
    print(f"Previous action : {prev_action}")
    print(f"Q before        : {q_before:.6f}")

    # Improve observed outcome before second decision.
    # This creates a positive reward and triggers Q-table update.
    state.delivery_estimate = 1.00
    state.d_hat = 0.05
    state.l_hat = 0.50
    state.u_hat = 0.50
    state.rho_hat = 0.02

    q.decide_mode_and_fanout(state)

    q_after = q.q_table[prev_state][prev_action]
    summary = q.get_learning_summary()

    print(f"Q after         : {q_after:.6f}")
    print(f"Q updates       : {summary['q_updates']}")
    print(f"Mean reward     : {summary['q_mean_reward']:.6f}")

    print("\nRESULT")
    if summary["q_updates"] >= 1 and q_after != q_before:
        print("PASS: Q-value changed after reward and next-state observation.")
    else:
        print("FAIL: Q-value did not change.")


if __name__ == "__main__":
    main()