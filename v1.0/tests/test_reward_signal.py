from ahbn.control import AHBNController, AHBNParams, NodeControlState
from ahbn.q_learning import QAHBNController


def main():

    params = AHBNParams()
    base = AHBNController(params)

    q = QAHBNController(
        base_controller=base,
        cfg={},
        seed=42,
    )

    print("\n==============================")
    print("REWARD SIGNAL TEST")
    print("==============================")

    # ----------------------------------
    # Poor network condition
    # ----------------------------------

    bad_state = NodeControlState(
        d_hat=0.60,
        u_hat=4.0,
        l_hat=3.0,
        rho_hat=0.40,
        r_hat=0.60,
        c_hat=0.50,
        delivery_estimate=0.20,
    )

    bad_reward = q.compute_reward(bad_state)

    print("\nBAD STATE")
    print(f"Reward = {bad_reward:.3f}")

    # ----------------------------------
    # Good network condition
    # ----------------------------------

    good_state = NodeControlState(
        d_hat=0.05,
        u_hat=0.50,
        l_hat=0.50,
        rho_hat=0.02,
        r_hat=0.05,
        c_hat=0.05,
        delivery_estimate=1.00,
    )

    good_reward = q.compute_reward(good_state)

    print("\nGOOD STATE")
    print(f"Reward = {good_reward:.3f}")

    print("\nRESULT")

    if good_reward > bad_reward:
        print("PASS: Better network condition produces higher reward.")
    else:
        print("FAIL: Reward signal is inverted.")


if __name__ == "__main__":
    main()