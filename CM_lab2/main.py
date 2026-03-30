import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return x * np.sin(x)

def f2(x):
    return np.sin(np.abs(x)) + 1

def df1(x):
    return 2 * np.cos(x) - x * np.sin(x)

def df2(x):
    if x > 0:
        return -np.sin(x)
    elif x < 0:
        return -np.sin(-x)
    else:
        return 0

def separated_differences(x, func):
    n = len(x)
    sd = np.zeros((n, n))
    y = func(x)
    for i in range(n):
        sd[i, 0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            sd[i, j] = (sd[i + 1, j - 1] - sd[i, j - 1]) / (x[i + j] - x[i])
    return sd

def bracket(xi):
    if xi > 0:
        return f"(x - {xi})"
    elif xi < 0:
        return f"(x + {abs(xi)})"
    else:
        return "x"

def newton_polynomial_2(x, sd):
    a0, a1, a2 = sd[0, 0], sd[0, 1], sd[0, 2]
    sign1 = "+" if a1 >= 0 else "-"
    sign2 = "+" if a2 >= 0 else "-"
    a1 = abs(a1)
    a2 = abs(a2)
    term1 = bracket(x[0])
    term2 = bracket(x[1])
    print("Аналитическое представление многочлена Ньютона 2-й степени:")
    print(f"P2(x) = {a0} {sign1} {a1}{term1} {sign2} {a2}{term1}{term2}")

def newton_value(x, sd, t):
    n = len(x)
    result = sd[0, 0]
    coef_x = 1.0
    for i in range(1, n):
        coef_x *= (t - x[i - 1])
        result += sd[0, i] * coef_x
    return result

def chebyshev_nodes(a, b, n):
    n += 1
    return np.array([
        (a + b) / 2 + ((b - a) / 2) * np.cos(np.pi * (2 * k + 1) / (2 * n))
        for k in range(n)
    ])

def findMaxInRow(row, colon_num):
    maxInd = colon_num
    for j in range(colon_num, len(row)):
        if np.abs(row[j]) > np.abs(row[maxInd]):
            maxInd = j
    return maxInd

def gauss(n, aC, bC):
    a = aC.copy().astype(float)
    b = bC.copy().astype(float)
    for m in range(n - 1):
        j = findMaxInRow(a[m], m)
        if j != m:
            a[:, [m, j]] = a[:, [j, m]]
        for k in range(m + 1, n):
            l = a[k][m] / a[m][m]
            for j in range(m, n):
                a[k][j] -= l * a[m][j]
            b[k] -= l * b[m]
    x = np.zeros(n)
    for i in reversed(range(n)):
        s = 0.0
        for j in range(i + 1, n):
            s += a[i][j] * x[j]
        x[i] = (b[i] - s) / a[i][i]
    return x

def spline(x, y, sd, n, m0, mn):
    h = np.zeros(n - 1)
    for i in range(n - 1):
        h[i] = x[i + 1] - x[i]
    A = np.zeros((n, n))
    b = np.zeros(n)
    A[0][0] = 1
    b[0] = m0
    for i in range(1, n - 1):
        A[i][i - 1] = h[i - 1]
        A[i][i] = 2 * (h[i - 1] + h[i])
        A[i][i + 1] = h[i]
        b[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    A[n - 1][n - 1] = 1
    b[n - 1] = mn
    gamma = gauss(n, A, b)
    splineCoef = np.zeros((n - 1, 4))
    for i in range(n - 1):
        h_i = x[i + 1] - x[i]
        splineCoef[i][0] = y[i]
        splineCoef[i][1] = (y[i + 1] - y[i]) / h_i - h_i * (2 * gamma[i] + gamma[i + 1]) / 6
        splineCoef[i][2] = gamma[i] / 2
        splineCoef[i][3] = (gamma[i + 1] - gamma[i]) / (6 * h_i)
    return splineCoef

def findSplineIndex(a, b, t, n):
    h = (b - a) / (n - 1)
    index = int((t - a) // h)
    if index >= n - 1:
        index = n - 2
    if index < 0:
        index = 0
    return index

def splineValue(coef, x, t, a, b):
    i = findSplineIndex(a, b, t, len(x))
    dx = t - x[i]
    return coef[i][0] + coef[i][1] * dx + coef[i][2] * dx**2 + coef[i][3] * dx**3

n = 16
a = -2
b = 2

x = np.arange(a, b, 0.01)

f = f2
df = df2

nodes = np.linspace(a, b, n + 1)

sd = separated_differences(nodes, f)

spline_ = spline(nodes, f(nodes), sd, n + 1, df(a), df(b))

spline_values = [splineValue(spline_, nodes, t, a, b) for t in x]

plt.title(f'n = {n}')
plt.plot(x, spline_values, label='S(x)')
plt.plot(x, f(x), label='y = sin(|x|) + 1')
plt.legend()
plt.show()