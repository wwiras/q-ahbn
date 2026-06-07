from ahbn.control import AHBNController, AHBNParams, NodeControlState
from ahbn.q_learning import QAHBNController


def main():

    params = AHBNParams()
    base = AHBNController(params)

    q = QAHBNController(
        base_controller=base,
        cfg={
            "epsilon": 0.0,      # force exploitation
            "epsilon_min": 0.0,
            "epsilon_decay": 1.0,
        },
        seed=42,
    )

    print("\n==============================")
    print("ACTION SELECTION TEST")
    print("==============================")

    state = NodeControlState(
        d_hat=0.20,
        u_hat=1.00,
        l_hat=1.00,
        rho_hat=0.05,
        r_hat=0.10,
        c_hat=0.10,
        delivery_estimate=0.50,
    )

    s = q.discretize_state(state)

    print(f"\nState = {s}")

    # --------------------------------------------------
    # Create known Q-values
    # --------------------------------------------------

    q.q_table[s]["ahbn_base"] = 0.50
    q.q_table[s]["more_structured"] = 1.20
    q.q_table[s]["more_gossip"] = 3.50
    q.q_table[s]["duplicate_suppression"] = 0.80
    q.q_table[s]["recovery_push"] = 2.10
    q.q_table[s]["resource_conservative"] = 0.30

    print("\nQ-values")

    for action, value in q.q_table[s].items():
        print(f"{action:24s} = {value:.3f}")

    selected = q.choose_action(s)

    print(f"\nSelected Action = {selected}")

    print("\nRESULT")

    if selected == "more_gossip":
        print("PASS: Policy selected highest-Q action.")
    else:
        print("FAIL: Policy did not select highest-Q action.")


if __name__ == "__main__":
    main()