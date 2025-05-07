import numpy as np
import matplotlib.pyplot as plt

max_iter = 500
n = 100

exact_line_xk = []
backtrack_line_xk = []

def GradientDescent(Q, q):
    x_star = -np.linalg.solve(Q, q)
    p = f(Q, q, x_star)
    tol = 1e-4
    n = Q.shape[1]  # dimensions of x
    x = np.zeros((n, 1))   # initialize x as a zero vector with n dimensions
    for i in range(max_iter):
        exact_line_xk.append((f(Q, q, x) - p).item())
        g = Gradient(Q, q, x)
        if np.linalg.norm(g) <= tol:
            break
        delta_x = -g
        t = ExactLineSearch(g, Q)
        x = x + t * delta_x
    print("iterations needed: ", i+1)
    print("optimal value: ", p)
    print("Gradient Descent value: ", f(Q, q, x))
    print("||x - x_star||: ", np.linalg.norm(x - x_star))
    return x


def GradientDescentBacktracking(Q, q):
    x_star = -np.linalg.solve(Q, q)
    p = f(Q, q, x_star)
    tol = 1e-4
    n = Q.shape[1]  # dimensions of x
    x = np.zeros((n, 1))   # initialize x as a zero vector with n dimensions
    for i in range(max_iter):
        backtrack_line_xk.append((f(Q, q, x) - p).item())
        g = Gradient(Q, q, x)
        if np.linalg.norm(g) <= tol:
            break
        delta_x = -g
        t = BacktrackingLineSearch(g, Q, q, x)
        x = x + t * delta_x
    print("\niterations needed: ", i+1)
    print("optimal value: ", p)
    print("Gradient Descent value: ", f(Q, q, x))
    print("||x - x_star||: ", np.linalg.norm(x - x_star))
    return x


def f(Q, q, x):
    return 0.5 * x.T @ Q @ x + q.T @ x


def Gradient(Q, q, x):
    return Q @ x + q


def ExactLineSearch(g, Q):
    n = g.T @ g
    d = g.T @ Q @ g
    return (n/d).item()


def BacktrackingLineSearch(g, Q, q, x, alpha=0.2, beta=0.5):
    t = 1
    while f(Q, q, x + t * -g) > f(Q, q, x) + alpha * t * g.T @ -g:
        t *= beta
    return t    


def MakePDMatrix(n, condition_target):
    A = np.random.randn(n,n)
    s = np.linalg.svd(A, compute_uv=False)
    sigma_max = s[0]**2
    sigma_min = s[-1]**2
    lambdal = (sigma_max - condition_target * sigma_min) / (condition_target - 1)
    lambdal = max(lambdal, 0)
    return A.T @ A + lambdal * np.eye(n)

def PlotLines():
    plt.plot(exact_line_xk, label='Exact Line Search')
    plt.plot(backtrack_line_xk, label='Backtracking Line Search')
    plt.xlabel('k')
    plt.ylabel('f(x_k) - p*')
    plt.title('Gradient Descent with Exact and Backtracking Line Search')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.legend()
    plt.show()

def main():
    Q = MakePDMatrix(n, 5)
    q = np.random.randn(n,1)
    GradientDescent(Q, q)
    GradientDescentBacktracking(Q, q)
    PlotLines()
    
if __name__ == "__main__":
    main()