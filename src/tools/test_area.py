from elements.quantum.spin import Spin

def main():
    proj1 = Spin.z_plus @ Spin.z_plus.dagger
    proj2 = Spin.x_plus @ Spin.x_plus.dagger
    print(0.5 * proj1 + 0.5 * proj2)


if __name__ == "__main__":
    main()
