
def quick_pow(base, exp, mod) -> (str, int):
    binary_number = f"{exp:0b}"
    res = f"{exp} = {binary_number}\n"

    number_of_bits = len(f"{exp:0b}")

    res += f"{base}^{exp} mod {mod} =\n"

    a = base
    for i in range(number_of_bits - 1):
        res += f"= {'(' * (number_of_bits - 1 - i)}{a}"
        for j in range(i, number_of_bits):
            if j != i:
                res += " * "
                res += str(base) if binary_number[j] == "1" else "1"

            if j != number_of_bits - 1:
                res += ")^2"

        res += f" mod {mod} =\n"

        a = ((a ** 2) * (base ** int(binary_number[i + 1]))) % mod

    res += f"= {a} mod {mod}\n"

    return res, a


if __name__ == '__main__':
    res, num = quick_pow(175, 235, 257)
    print(res, num)