import torch
from tqdm import tqdm

def get_trainable_params(model):
    return list(filter(lambda p: p.requires_grad, model.parameters()))

def conjugate_gradient(hvp_fn, b, tol=1e-5, max_iter=50):
    """
    Conjugate Gradient method to solve Ax = b, where A is represented by hvp_fn.
    Args:
        hvp_fn: Function to compute Hessian-vector product, i.e., A*v.
        b: Right-hand side vector in Ax = b.
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.
    Returns:
        x: Solution vector.
    """
    x = torch.zeros_like(b)  # Initial guess x_0 = 0
    r = b.clone()            # Initial residual r_0 = b - A*x_0 = b
    p = r.clone()            # Initial search direction p_0 = r_0
    rs_old = torch.dot(r, r)

    for i in tqdm(range(max_iter), desc="Conjugate Gradient", leave=False):
        Ap = hvp_fn(p)       # Compute A*p
        alpha = rs_old / torch.dot(p, Ap)  # Step size
        x += alpha * p       # Update x
        r -= alpha * Ap      # Update residual
        rs_new = torch.dot(r, r)

        if torch.sqrt(rs_new) < tol:  # Convergence check
            break

        beta = rs_new / rs_old
        p = r + beta * p      # Update search direction
        rs_old = rs_new

    return x
