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
    # Можно самим реализовать алгоритм в выводом TODO

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
            # if d is None: #(e, fn) = 1 TODO

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
    res = f"Расшифрование данных алгоритмом RSA:\n"
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
    res = f"Подпись сообщения алгоритмом RSA:\n"
    c = None

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

        s, c = quick_pow(m, d, n)
        res += s
    except TypeError:
        res += "Недостаточно данных\n"
    else:
        res += f"s = m^d mod n = {m}^{d} mod {n} = {c} mod {n}\n"

    return res, c


def rsa_check_sign(c=None, e=None, d=None, n=None, p=None, q=None, fn=None) -> (str, int):
    res = f"Проверка подписи алгоритмом RSA:\n"
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


def el_gamal_encrypt(m=None, p=None, g=None, y=None, x=None, k = None) -> (str, int, int):
    res = f"Шифрование данных алгоритмом Эль Гамаля:\n"
    res+= f"a = g^k mod p\n"
    res+= f"a = {g}^{k} mod {p}\n"
    s, a = quick_pow(g, k, p)
    res+= f"b = m * y^k mod p\n"
    res+= f"b = {m} * {y}^{k} mod {p}\n"
    s, b = quick_pow(y, k, p)
    b = b * m % p
    res+=f"(a, b) = ({a}, {b})\n"

    return res, a, b


def el_gamal_decrypt(p=None, g=None, y=None, x=None, a=None, b=None) -> (str, int):
    res = f"Дешифрование данных алгоритмом Эль Гамаля:\n"
    res+= f"message = (b / a^x) mod p\n"
    res+= f"message = ({b} / a^{x}) mod {p}\n"
    s, a_pow_x = quick_pow(a, x, p)
    res+=s
    s, m = solve_congruence(a_pow_x, b, p, "message")
    res+=s
    res+= f"message = {m}\n"
    return res, m


def el_gamal_sign(m=None, p=None, g=None, y=None, x=None, k=None) -> (str, int, int, int):
    res = f"Подпись сообщения алгоритмом Эль Гамаля:\n"

    if y == None:
        res+=f"y = {g}^{x} mod {p}\n"
        s, y = quick_pow(g, x, p)
        res+=s
    
    res+=f"r = g^k mod p\n"
    res+=f"r = {g}^{k} mod {p}\n"
    s, r = quick_pow(g, k, p)
    res+=s
    res+=f"sig = (m - x*r/k) mod (p - 1)\n"
    res+=f"sig = ({m - x*r}/{k}) mod {p - 1}\n"
    s, sig = solve_congruence(k, m - x*r, p - 1)
    res+=s
    res+= f"Подписанное сообщение <{m}, {r}, {sig}>\n"

    return res, m, r, sig


def el_gamal_check_sign(m=None, p=None, g=None, y=None, r=None, sig=None) -> str:
    res = f"Проверка подписи алгоритмом Эль Гамаля:\n"
    res+= f"Подписанное сообщение <{m}, {r}, {sig}>, открытый ключ y={y}\n"
    res+=f"Проверяем сравнение y^r * r^s mod p = g^m mod p\n"
    res+=f"{y}^{r} * {r}^{sig} mod {p} ? {g}^{m} mod {p}\n"
    s, y_pow_r = quick_pow(y, r, p)
    s, r_pow_sig = quick_pow(r, sig, p)
    left_value = y_pow_r*r_pow_sig % p
    s, right_value = quick_pow(g, m, p)
    res+=f"{y_pow_r} * {r_pow_sig} mod {p} ? {right_value} mod {p}\n"
    res+=f"{left_value} mod {p} ? {right_value} mod {p}\n"
    if (left_value == right_value):
        res+=f"Подпись верна\n"
    else:
        res+=f"Подпись неверна\n"
    
    return res


def miller_test(a, n) -> (str, bool):
    res = f"Тест Миллера на простоту числа n = {n} со свидетелем простоты a = {a}:\n"
    res += "n - 1 = r * 2 ^ s\n"

    r = n - 1
    s = 0
    while r % 2 == 0:
        r //= 2
        s += 1

    res += f"{n} - 1 = {r} * 2 ^ {s}\n"

    primality = False
    for i in range(s + 1):
        s, x = quick_pow(a, (2 ** i) * r, n)
        res += f"a^(2^{i} * {r}) mod n = {x}\n"
        res += s

        if x == n - 1:
            res += f"В ряду есть -1 (все остальные 1) => a = {a} свидетель простоты числа n = {n}\n"
            primality = True
            break

    if not primality:
        res += f"Число n = {n} - составное (a = {a} не является свидетелем его простоты)\n"

    return res, primality


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
    output, a, b = el_gamal_encrypt(m=5, p=29, g=10, y=8, x=5, k=5)
    print(output)
    output, num = el_gamal_decrypt(p=23, g=5, y=9, x=10, a=10, b=18)
    print(output)
    output, m, r, sig = el_gamal_sign(m=3, p=23, g=5, y=17, x=7, k=5)
    print(output)
    output = el_gamal_check_sign(m=3, p=23, g=5, y=17, r=20, sig=21)
    print(output)
    output, primality = miller_test(a=104, n=145)
    print(output)


if __name__ == '__main__':
    test()
