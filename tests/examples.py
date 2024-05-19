import numpy as np

def quadratic_function_1(x, hessian_evaluation_needed=False):
    Q = np.array([[1, 0], [0, 1]])
    f = x.T @ Q @ x
    g = 2 * Q @ x
    if hessian_evaluation_needed:
        h = 2 * Q
        return f, g, h
    return f, g, None

def quadratic_function_2(x, hessian_evaluation_needed=False):
    Q = np.array([[1, 0], [0, 100]])
    f = x.T @ Q @ x
    g = 2 * Q @ x
    if hessian_evaluation_needed:
        h = 2 * Q
        return f, g, h
    return f, g, None

def quadratic_function_3(x, hessian_evaluation_needed=False):
    #data preperation
    Q_middle = np.array([[100, 0], [0, 1]])
    Q_main = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    Q_calc = Q_main.T @ Q_middle @ Q_main

    #calculations
    f = x.T @ Q_calc @ x
    g = 2 * Q_calc @ x
    if hessian_evaluation_needed:
        h = 2 * Q_calc
        return f, g, h
    return f, g, None

def rosenbrock_function(x, hessian_evaluation_needed=False):
    x1, x2 = x[0], x[1]
    f = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    g = np.array([
        -400 * x1 * (x2 - x1**2) - 2 * (1 - x1),
        200 * (x2 - x1**2)
    ])
    if hessian_evaluation_needed:
        h = np.array([
            [1200 * x1**2 - 400 * x2 + 2, -400 * x1],
            [-400 * x1, 200]
        ])
        return f, g, h
    return f, g, None

def linear_function(x, hessian_evaluation_needed=False):
    a = np.array([1.0, 2.0])
    f = a.T @ x
    g = a
    if hessian_evaluation_needed:
        h = np.zeros((len(x), len(x)))
        return f, g, h
    return f, g, None

def exponential_function(x, hessian_evaluation_needed=False):
    x1, x2 = x[0], x[1]
    f = np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) + np.exp(-x1 - 0.1)
    g = np.array([
        np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) - np.exp(-x1 - 0.1),
        3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1)
    ])
    if hessian_evaluation_needed:
        h = np.array([
            [np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) + np.exp(-x1 - 0.1), 
             3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1)],
            [3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1),
             9 * np.exp(x1 + 3 * x2 - 0.1) + 9 * np.exp(x1 - 3 * x2 - 0.1)]
        ])
        return f, g, h
    return f, g, None
