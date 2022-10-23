import math


def solve_linear_congruence(a, b, m) -> (int, int):
    g = math.gcd(a, m)

    if b % g:
        raise ValueError("Решений нет")

    a, b, m = a//g, b//g, m//g

    return pow(a, -1, m) * b % m, m


def solve_congruence(a, b, m, var_name="x") -> (str, int):
    """Solve ax = b mod m"""

    res = f"Решение сравнения: {a} * {var_name} = {b} mod {m}\n"
    #Можно самим реализовать алгоритм в выводом TODO

    try:
        x, mx = solve_linear_congruence(a, b, m)
    except ValueError:
        res += "Решения нет\n"
    else:
        res += f"Решение: {var_name} = {x} mod {mx}\n"

    return res, x


def quick_pow(base, exp, mod) -> (str, int):
    res = f"Быстрое возведение в степень {base}^{exp} mod {mod}:\n"
    binary_number = f"{exp:0b}"
    res += f"{exp} (в десятичной сс) = {binary_number} (в двоичной сс)\n"

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


def rsa_encrypt(m=None, e=None, d=None, n=None, p=None, q=None, fn=None) -> (str, int):
    res = f"Шифрование данных алгоритмом RSA:\n"
    c = None

    try:
        if n is None:
            n = p * q
            res += f"n = p * q = {p} * {q} = {n}\n"

        if e is None:
            #if d is None: #(e, fn) = 1 TODO

            if fn is None:
                fn = (p - 1) * (q - 1)
                res += f"ф(n) = (p-1)*(q-1) = {p-1}*{q-1} = {fn}\n"

            s, e = solve_congruence(d, 1, fn, "e")
            res += s

        s, c = quick_pow(m, e, n)
        res += s
    except TypeError:
        res += "Недостаточно данных\n"
    else:
        res += f"c = m^e mod n = {m}^{e} mod {n} = {c} mod {n}\n"

    return res, c


def rsa_decrypt(c=None, e=None, d=None, n=None, p=None, q=None, fn=None) -> (str, int):
    res = f"Расифрование данных алгоритмом RSA:\n"
    m = None

    try:
        if n is None:
            n = p * q
            res += f"n = p * q = {p} * {q} = {n}\n"

        if d is None:
            # if e is None: #(e, fn) = 1 TODO

            if fn is None:
                fn = (p - 1) * (q - 1)
                res += f"ф(n) = (p-1)*(q-1) = {p - 1}*{q - 1} = {fn}\n"

            s, d = solve_congruence(e, 1, fn, "d")
            res += s

        s, m = quick_pow(c, d, n)
        res += s
    except TypeError:
        res += "Недостаточно данных\n"
    else:
        res += f"m = c^d mod n = {c}^{d} mod {n} = {m} mod {n}\n"

    return res, m


def rsa_sign(m=None, e=None, d=None, n=None, p=None, q=None, fn=None) -> (str, int):
    res = f"Шифрование данных алгоритмом RSA:\n"
    c = None

    try:
        if n is None:
            n = p * q
            res += f"n = p * q = {p} * {q} = {n}\n"

        if d is None:
            #if e is None: #(e, fn) = 1 TODO

            if fn is None:
                fn = (p - 1) * (q - 1)
                res += f"ф(n) = (p-1)*(q-1) = {p - 1}*{q - 1} = {fn}\n"

            s, d = solve_congruence(e, 1, fn, "d")
            res += s

        s, c = quick_pow(m, d, n)
        res += s
    except TypeError:
        res += "Недостаточно данных\n"
    else:
        res += f"s = m^d mod n = {m}^{d} mod {n} = {c} mod {n}\n"

    return res, c


def rsa_check_sign(c=None, e=None, d=None, n=None, p=None, q=None, fn=None) -> (str, int):
    res = f"Расифрование данных алгоритмом RSA:\n"
    m = None

    try:
        if n is None:
            n = p * q
            res += f"n = p * q = {p} * {q} = {n}\n"

        if e is None:
            # if d is None: #(e, fn) = 1 TODO

            if fn is None:
                fn = (p - 1) * (q - 1)
                res += f"ф(n) = (p-1)*(q-1) = {p - 1}*{q - 1} = {fn}\n"

            s, e = solve_congruence(d, 1, fn, "e")
            res += s

        s, m = quick_pow(c, e, n)
        res += s
    except TypeError:
        res += "Недостаточно данных\n"
    else:
        res += f"m\' = s^e mod n = {c}^{e} mod {n} = {m} mod {n}\n"

    return res, m


def test():
    output, num = quick_pow(175, 235, 257)
    print(output)
    output, num = solve_congruence(7, 2, 10)
    print(output)
    output, num = rsa_encrypt(p=7, q=11, m=29, d=7)
    print(output)
    output, num = rsa_decrypt(p=7, q=11, e=43, c=19)
    print(output)
    output, num = rsa_sign(p=5, q=11, d=3, m=40)
    print(output)
    output, num = rsa_check_sign(p=5, q=11, d=3, c=35)
    print(output)


if __name__ == '__main__':
    test()