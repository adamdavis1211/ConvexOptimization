import numpy as np
import matplotlib.pyplot as plt

max_iter = 200

gradient_descent = []
gradient_descent_backtracking = []
steepest_descent = []

def GradientDescent(Q, q):
    x_star = -np.linalg.solve(Q, q)
    p = f(Q, q, x_star)
    tol = 1e-4
    n = Q.shape[1]  # dimensions of x
    x = np.random.randn(n, 1)   # random numbers for components of x
    for i in range(max_iter):
        gradient_descent.append((f(Q, q, x) - p).item())
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


def GradientDescentBacktracking(Q, q,):
    x_star = -np.linalg.solve(Q, q)
    p = f(Q, q, x_star)
    tol = 1e-4
    n = Q.shape[1]  # dimensions of x
    x = np.random.randn(n, 1)   # random numbers for components of x
    for i in range(max_iter):
        gradient_descent_backtracking.append((f(Q, q, x) - p).item())
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


def SteepestDescent(Q, q, P):
    x_star = -np.linalg.solve(Q, q)
    p = f(Q, q, x_star)
    tol = 1e-4
    n = Q.shape[1]  # dimensions of x
    x = np.random.randn(n, 1)   # random numbers for components of x
    for i in range(max_iter):
        steepest_descent.append((f(Q, q, x) - p).item())
        g = Gradient(Q, q, x)
        if np.linalg.norm(g) <= tol:
            break
        delta_x = SteepestDescentStep(P, g)
        t = SteepestExactLineSearch(g, delta_x, Q)
        x = x + t * delta_x
    print("\niterations needed: ", i+1)
    print("optimal value: ", p)
    print("Steepest Descent value: ", f(Q, q, x))
    print("||x - x_star||: ", np.linalg.norm(x - x_star))
    return x
    
    
def SteepestExactLineSearch(g, d, Q):
    return - (g.T @ d / (d.T @ Q @ d)).item()


def f(Q, q, x):
    return 0.5 * x.T @ Q @ x + q.T @ x


def Gradient(Q, q, x):
    return Q @ x + q


def ExactLineSearch(g, Q):
    n = g.T @ g
    d = g.T @ Q @ g
    return (n/d).item()


def BacktrackingLineSearch(g, Q, q, x, alpha=0.4, beta=0.8):
    t = 1
    while f(Q, q, x + t * -g) > f(Q, q, x) + alpha * t * g.T @ -g:
        t *= beta
    return t    


def MakePDMatrix(n, condition):
    A = np.random.randn(n,n)
    s = np.linalg.svd(A, compute_uv=False)
    sigma_max = s[0]**2
    sigma_min = s[-1]**2
    lambdal = (sigma_max - condition * sigma_min) / (condition - 1)
    lambdal = max(lambdal, 0)
    return A.T @ A + lambdal * np.eye(n)

def SteepestDescentStep(P, grad):
    return -np.linalg.solve(P, grad)

def PlotLines():
    plt.plot(gradient_descent, label='Gradient Descent Exact Line Search')
    plt.plot(gradient_descent_backtracking, label='Gradient Descent Backtracking Line Search')
    plt.plot(steepest_descent, label='Steepest Descent')
    plt.xlabel('k')
    plt.ylabel('f(x_k) - p*')
    plt.title('Gradient Descent vs Steepest Descent')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.legend()
    plt.show()


def main():
    n = 500
    Q = MakePDMatrix(n, 15)
    P = np.diag(np.diag(Q))
    q = np.random.randn(n,1)
    GradientDescent(Q, q)
    GradientDescentBacktracking(Q, q)
    SteepestDescent(Q, q, P)
    PlotLines()
   
if __name__ == "__main__":
    main()