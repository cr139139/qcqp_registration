import dill
import numpy as np
import cvxpy as cp
import scipy as sp
import matplotlib.pyplot as plt

def ellipse_sdf(p, ab, iters=3):
    """
    Fast batched signed distance from 2D points to an ellipse using Newton iteration.
    """
    a, b = ab
    p_sign = np.sign(p)
    p_abs = np.abs(p)

    px, py = p_abs[:, 0], p_abs[:, 1]
    a2, b2 = a * a, b * b

    outside = (px * px) / a2 + (py * py) / b2 > 1.0

    # Initial angle guess
    w = np.where(
        outside,
        np.arctan2(py * a, px * b),
        np.where(a * (px - a) < b * (py - b), np.pi / 2, 0.0)
    )

    for _ in range(iters):
        cos_w = np.cos(w)
        sin_w = np.sin(w)

        ax, ay = a * cos_w, b * sin_w
        bx, by = -a * sin_w, b * cos_w

        dx = px - ax
        dy = py - ay

        dot_dv = dx * bx + dy * by
        dot_du = dx * ax + dy * ay
        dot_vv = bx * bx + by * by

        w += dot_dv / (dot_du + dot_vv)

    cos_w = np.cos(w)
    sin_w = np.sin(w)
    cx = a * cos_w
    cy = b * sin_w

    dx = px - cx
    dy = py - cy
    dist = np.sqrt(dx * dx + dy * dy)
    dist *= np.where(outside, 1.0, -1.0)

    closest = np.stack((cx, cy), axis=1) * p_sign
    return dist, closest


with open('asm_single.pkl', 'rb') as f:
    pca = dill.load(f)
B0 = pca.mean.reshape((-1, 2))  # 359x2
B = pca.components.T
B = B.reshape((-1, 2, B.shape[-1]))  # 359x2xc


X_init = B0
theta_init = np.random.uniform(low=0.0, high=np.pi * 2, size=None)
R_init = np.array([[np.cos(theta_init), -np.sin(theta_init)],
                   [np.sin(theta_init), np.cos(theta_init)]])
t_init = np.random.rand(2)

X = X_init @ R_init.T + t_init[None, :]

for i in range(100):
    D, Y = ellipse_sdf(X, np.array([0.5, 1.0]), iters=3)
    # Y = X_init
    plt.scatter(*X.T, color='r')
    plt.scatter(*Y.T, color='b')
    plt.axis('equal')
    plt.show()

    Q = np.eye(4+2+B.shape[-1])
    p = np.zeros(4+2+B.shape[-1])

    YY = Y[:, :, None] * Y[:, None, :]
    YY_mean = YY.mean(axis=0)
    Q[0:2, 0:2] = Q[2:4, 2:4] = YY_mean
    Q[4:5, 0:2] = Q[5:6, 2:4] = -Y.mean(axis=0, keepdims=True)
    Q[0:2, 4:5] = Q[2:4, 5:6] = -Y.mean(axis=0, keepdims=True).T
    BY = (B[:, :, None, :] * Y[:, None, :, None]).reshape((-1, 4, B.shape[-1]))
    Q[6:6+B.shape[-1], 0:4] = -BY.mean(axis=0).T
    Q[0:4, 6:6+B.shape[-1]] = -BY.mean(axis=0)
    Q[6:6+B.shape[-1], 4:6] = B.mean(axis=0).T
    Q[4:6, 6:6+B.shape[-1]] = B.mean(axis=0)
    BB = B[:, :, :, None] * B[:, :, None, :]
    Q[6:6+B.shape[-1], 6:6+B.shape[-1]] = BB.mean(axis=0).sum(axis=0)
    Q[6:6+B.shape[-1], 6:6+B.shape[-1]] += np.eye(B.shape[-1]) * 1e-4

    B0Y = (B0[:, :, None] * Y[:, None, :]).reshape((-1, 4))
    p[0:4] = -2 * B0Y.mean(axis=0)
    p[4:6] = 2 * B0.mean(axis=0)
    p[6:6+B.shape[-1]] = 2 * (B0[:, :, None] * B).sum(axis=1).mean(axis=0)

    # with np.printoptions(precision=3, suppress=True):
    #     print(Q)
    #     print(p)

    sol = np.linalg.solve(Q, -p/2)
    sol = np.linalg.inv(Q) @ (-p/2)
    R_temp = sol[:4].reshape((2, 2))

    U, _, Vt = np.linalg.svd(R_temp)
    R = U @ Vt
    det = np.linalg.det(R)
    Vt[1, 0:2] *= det
    R = U @ Vt
    s = 1 # (R_temp.transpose() @ R).trace() / 2.
    t = sol[4:6]
    c = sol[6:]

    R_real = R.T
    s_real = 1/s
    t_real = s_real * (R_real @ t)

    print(s_real)

    X = (B0 + (B @ c[None, :, None])[:, :, 0]) @ R_real.T * s_real + t_real



