import numpy as np
import matplotlib.pyplot as plt

newton_xk = []

def f(x1, x2):
    return np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) + np.exp(-x1 - 0.1)


def Gradient(x1, x2):
    return np.array([np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) - np.exp(-x1 - 0.1),
                     3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1)])


def Hessian(x1, x2):  
    return np.array([[np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) + np.exp(-x1 - 0.1),
                      3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1)],
                     [3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1),
                      9 * np.exp(x1 + 3 * x2 - 0.1) + 9 * np.exp(x1 - 3 * x2 - 0.1)]])


def NewtonStep(x1, x2):
    grad = Gradient(x1, x2)
    hess = Hessian(x1, x2)
    step = np.linalg.solve(hess, -grad)
    return step


def NewtonDecrement(grad, step):
    return -grad @ step

def BacktrackingLineSearch(step, grad, x, alpha=0.1, beta=0.7):
    t = 1
    while f(x[0] + t * step[0], x[1] + t * step[1]) > f(x[0], x[1]) + alpha * t * np.dot(grad, step):
        t *= beta
    return t

def NewtonMethod(x_init, tol=10e-15, max_iter=50):
    p = f(-0.5 * np.log(2), 0)
    x = np.array(x_init, dtype=float)  
    for i in range(max_iter):
        grad = Gradient(x[0], x[1])
        step = NewtonStep(x[0], x[1])
        lambda_sq = NewtonDecrement(grad, step)
        newton_xk.append((f(x[0], x[1]) - p).item())
        if lambda_sq / 2 <= tol:
            print(f"Converged in {i} iterations.")
            return x
        t = BacktrackingLineSearch(step, grad, x)
        x += t * step
    print("Reached maximum iterations.")
    return x

def PlotLines():
    plt.plot(newton_xk, label='Newton Method', color='blue')
    plt.xlabel('Iteration (k)')
    plt.ylabel('f(x_k) - p*')
    plt.title('Unconstrained Optimization with Newton Method')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.legend()
    plt.show()



def main():
    x_start = [0.0, 0.0]
    x_min = NewtonMethod(x_start)
    print("Minimizer found at:", x_min)
    print("Function value at minimizer:", f(x_min[0], x_min[1]))
    PlotLines()

if __name__ == "__main__":
    main()
