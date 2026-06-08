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
    print("STATE DISCRETIZATION TEST")
    print("==============================")

    # --------------------------------------------------
    # Low state
    # Expected:
    # D=L, L=L, U=L, C=L, R=L, Cap=L
    # --------------------------------------------------

    low_state = NodeControlState(
        d_hat=0.05,
        l_hat=0.50,
        u_hat=1.00,
        rho_hat=0.02,
        r_hat=0.10,
        c_hat=0.10,
    )

    low_key = q.discretize_state(low_state)

    print("\nLOW STATE")
    print(f"Mapped State = {low_key}")

    # --------------------------------------------------
    # Medium state
    # --------------------------------------------------

    medium_state = NodeControlState(
        d_hat=0.20,
        l_hat=2.00,
        u_hat=4.00,
        rho_hat=0.10,
        r_hat=0.40,
        c_hat=0.40,
    )

    medium_key = q.discretize_state(medium_state)

    print("\nMEDIUM STATE")
    print(f"Mapped State = {medium_key}")

    # --------------------------------------------------
    # High state
    # --------------------------------------------------

    high_state = NodeControlState(
        d_hat=0.60,
        l_hat=4.00,
        u_hat=8.00,
        rho_hat=0.40,
        r_hat=1.00,
        c_hat=0.90,
    )

    high_key = q.discretize_state(high_state)

    print("\nHIGH STATE")
    print(f"Mapped State = {high_key}")

    print("\nRESULT")

    expected_low = ("L", "L", "L", "L", "L", "L")
    expected_high = ("H", "H", "H", "H", "H", "H")

    if low_key == expected_low and high_key == expected_high:
        print("PASS: State discretization maps metrics correctly.")
    else:
        print("FAIL: Unexpected state mapping.")


if __name__ == "__main__":
    main()