from elements.quantum.spin import Spin


def main():
    print(Spin.z_plus())
    print(Spin.z_minus())
    print(Spin.x_plus())
    print("x+ norm:", Spin.x_plus().norm)
    print("<z+|z+>:", Spin.z_plus().conjugate_transpose() @ Spin.z_plus())
    print("<z-|z->:", Spin.z_minus().conjugate_transpose() @ Spin.z_minus())
    print("<z-|z+>:", Spin.z_minus().conjugate_transpose() @ Spin.z_plus())
    print("<z+|x+>:", Spin.z_plus().conjugate_transpose() @ Spin.x_plus())
    print("<z+|x->:", Spin.z_plus().conjugate_transpose() @ Spin.x_minus())


if __name__ == "__main__":
    main()
