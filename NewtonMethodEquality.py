import numpy as np
import matplotlib.pyplot as plt

max_iter = 500
n = 100
p = 30
x_hat = np.random.uniform(low=0.1, high=1.0, size=n)
x0 = np.ones(n)
mu0 = np.zeros(p)

def f(x):
    return np.sum(x * np.log(x))

def generate_full_rank_matrix(p, n):
    while True:
        A = np.random.randn(p, n)  # standard normal entries
        if np.linalg.matrix_rank(A) == p:
            return A

def Gradient(x):
    return np.log(x) + 1

def Hessian(x):
    return np.diag(1 / x)

A = generate_full_rank_matrix(p, n)
b = A @ x_hat

def NewtonMethodInfeasible(x, mu, A, b):
    epsilon = 1e-4 # tolerance for stopping criterion
    f_values = [f(x)]
    for i in range(max_iter):
        residual_dual, residual_primal = FindResiduals(x, mu, A, b)
        if np.linalg.norm(np.concatenate([residual_dual, residual_primal])) < epsilon:
            return x, f_values 
        x_nt, mu_nt = ComputeNewtonStep(x, mu, A, b)
        t = BacktrackingLineSearch(x, mu, x_nt, mu_nt, A, b)
        x = x + t * x_nt
        mu = mu + t * mu_nt
        f_values.append(f(x))
    print("Did not converge.")
    return x, f_values

def ComputeNewtonStep(x, mu, A, b):
    hess = Hessian(x)
    residual_dual, residual_primal = FindResiduals(x, mu, A, b)
    KKT_matrix = np.block([[hess, A.T], [A, np.zeros((A.shape[0], A.shape[0]))]])
    rd_rp_vector = -np.concatenate([residual_dual, residual_primal])
    delta = np.linalg.solve(KKT_matrix, rd_rp_vector)
    return delta[:n], delta[n:]

def FindResiduals(x, mu, A, b):
    grad = Gradient(x)
    return (A.T @ mu + grad, A @ x - b) 

def BacktrackingLineSearch(x, mu, x_nt, mu_nt, A, b, alpha=0.4, beta=0.8):
    t = 1
    while t > 1e-8:
        new_x = x + t * x_nt
        if np.any(new_x <= 0):  # enforce x ∈ R++^n
            t *= beta
            continue
        if np.linalg.norm(np.concatenate(FindResiduals(new_x, mu + t * mu_nt, A, b))) <= \
           (1 - alpha * t) * np.linalg.norm(np.concatenate(FindResiduals(x, mu, A, b))):
            break
        t *= beta
    return t
    
def main():
    b = A @ x_hat

    x_star1, f_values1 = NewtonMethodInfeasible(np.copy(x_hat), np.zeros(p), A, b)
    x_star2, f_values2 = NewtonMethodInfeasible(np.ones(n), np.zeros(p), A, b)

    print("Optimal point from x̂: ", x_star1)
    print("Optimal value of f(x) from x̂: ", f(x_star1))
    print("Optimal point from 1s: ", x_star2)
    print("Optimal value of f(x) from 1s: ", f(x_star2))

    plt.plot(f_values1, label="Start from x̂ (feasible)", marker='o')
    plt.plot(f_values2, label="Start from 1s (infeasible)", marker='x')
    plt.title("Convergence of f(x) during Newton's Method")
    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()