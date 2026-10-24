def split_sign(coefficient: complex) -> tuple[str, str]:
    # Adding 0.0 turns the -0.0 that rounding can leave behind into 0.0, so it never prints "-0.00".
    real = round(coefficient.real, 2) + 0.0
    imag = round(coefficient.imag, 2) + 0.0
    if imag == 0:
        return ("-" if real < 0 else "+"), f"{abs(real):.2f}"
    if real == 0:
        return ("-" if imag < 0 else "+"), f"{abs(imag):.2f}j"
    return "+", f"({real:.2f}{imag:+.2f}j)"
