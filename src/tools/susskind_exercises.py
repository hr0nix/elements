from elements.quantum.spin import Spin, TwoSpins
import pytest


def exercise_6_6():
    singlet = TwoSpins.singlet
    operator = Spin.sigma_x.kronecker_product(Spin.sigma_y)
    expectation_value = singlet.dagger @ operator @ singlet
    assert expectation_value.real == pytest.approx(0.0)
    assert expectation_value.imag == pytest.approx(0.0)


def exercise_6_7_and_6_8():
    for triplet_name, triplet in (
        ("triplet_1", TwoSpins.triplet_1),
        ("triplet_2", TwoSpins.triplet_2),
        ("triplet_3", TwoSpins.triplet_3)
    ):
        for operator_name, operator in (
            ("x", Spin.sigma_x),
            ("y", Spin.sigma_y),
            ("z", Spin.sigma_z)
        ):
            operator_squared = operator.kronecker_product(operator)
            expectation_value = triplet.dagger @ operator_squared @ triplet
            print(f"Expectation value of sigma_{operator_name} x sigma_{operator_name} for {triplet_name}: {expectation_value.real}")


def exercise_6_9():
    sigma_tau = (
        Spin.sigma_x.kronecker_product(Spin.sigma_x) +
        Spin.sigma_y.kronecker_product(Spin.sigma_y) +
        Spin.sigma_z.kronecker_product(Spin.sigma_z)
    )
    assert (sigma_tau @ TwoSpins.singlet).close_to(-3.0 * TwoSpins.singlet)
    assert (sigma_tau @ TwoSpins.triplet_1).close_to(1.0 * TwoSpins.triplet_1)
    assert (sigma_tau @ TwoSpins.triplet_2).close_to(1.0 * TwoSpins.triplet_2)
    assert (sigma_tau @ TwoSpins.triplet_3).close_to(1.0 * TwoSpins.triplet_3)


if __name__ == "__main__":
    exercise_6_7_and_6_8()
