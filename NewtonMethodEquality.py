import numpy as np

n = 100
p = 30
x_hat = np.random.uniform(low=0.0, high=1.0, size=n)


def f(x):
    np.sum(x * np.log(x))

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


print(f"Shape of A: {A.shape}")
print(f"Rank of A: {np.linalg.matrix_rank(A)}")