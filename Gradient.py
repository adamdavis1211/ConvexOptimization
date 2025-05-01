import numpy as np


max_iter = 200


def GradientDescent(Q, q):
    x_star = -np.linalg.solve(Q, q)
    p = f(Q, q, x_star)
    tol = 1e-4
    n = Q.shape[1]  # dimensions of x
    x = np.random.randn(n, 1)   # random numbers for components of x
    for i in range(max_iter):
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
    x = np.random.randn(n, 1)   # random numbers for components of x
    for i in range(max_iter):
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


def BacktrackingLineSearch(g, Q, q, x, alpha=0.4, beta=0.8):
    t = 1
    while f(Q, q, x + t * -g) > f(Q, q, x) + alpha * t * g.T @ -g:
        t *= beta
    return t    


def MakePDMatrix(n):
    A = np.random.randn(n,n)
    return A.T @ A + n * np.eye(n)

def SteepestDescent():

def main():
    n = 100
    Q = MakePDMatrix(n)
    
    q = np.random.randn(n,1)
    GradientDescent(Q, q)
    GradientDescentBacktracking(Q, q)
   
if __name__ == "__main__":
    main()