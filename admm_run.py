import dill
import numpy as np
import cvxpy as cp
import scipy as sp

# np.random.seed(0)

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


with open('asm_double.pkl', 'rb') as f:
    pca = dill.load(f)
B0 = pca.mean.reshape((-1, 2))  # 359x2
B = pca.components.T
B = B.reshape((-1, 2, B.shape[-1]))  # 359x2xc


X_init = B0
theta_init = np.random.uniform(low=0.0, high=np.pi * 2, size=None)
R_init = np.array([[np.cos(theta_init), -np.sin(theta_init)],
                   [np.sin(theta_init), np.cos(theta_init)]])
t_init = np.random.rand(2)


# X = X_init @ R_init.T + t_init[None, :]
X = X_init
Y = (X_init) @ R_init.T + t_init[None, :]
D, Y = ellipse_sdf(X, np.ones(2) * 0.5, iters=3)

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

# rot+trans+asm+scale+slack
Q_new = np.eye(4+2+B.shape[-1]+2) * 0.
Q_new[0:4+2+B.shape[-1], 0:4+2+B.shape[-1]] = Q
Q_new[4+2+B.shape[-1]+1:4+2+B.shape[-1]+2, 0:4+2+B.shape[-1]] = p / 2.
Q_new[0:4+2+B.shape[-1], 4+2+B.shape[-1]+1:4+2+B.shape[-1]+2] = p[None].T / 2.
Q_new[-1, -1] = (B0**2).sum(axis=1).mean()

A0 = np.zeros((4+2+B.shape[-1]+2, 4+2+B.shape[-1]+2))
A0[-1, -1] = 1.

A = []

# w11^2 + w12^2 = s^2
A_temp = np.zeros((4+2+B.shape[-1]+2, 4+2+B.shape[-1]+2))
A_temp[[0, 1, -2], [0, 1, -2]] = np.array([1, 1, -1])
A.append(A_temp)
# w21^2 + w22^2 = s^2
A_temp = np.zeros((4+2+B.shape[-1]+2, 4+2+B.shape[-1]+2))
A_temp[[2, 3, -2], [2, 3, -2]] = np.array([1, 1, -1])
A.append(A_temp)
# w11 * w21 + w12 * w22 = 0
A_temp = np.zeros((4+2+B.shape[-1]+2, 4+2+B.shape[-1]+2))
A_temp[[0, 2, 1, 3], [2, 0, 3, 1]] = np.array([1, 0, 1, 0])
A.append(A_temp)
# w11 * w22 - w12 * w21 = s^2
A_temp = np.zeros((4+2+B.shape[-1]+2, 4+2+B.shape[-1]+2))
A_temp[[0, 1, 3, 2, -2], [3, 2, 0, 1, -2]] = np.array([1, -1, 1, -1, -2])
A.append(A_temp)
# s^2 = w^2
A_temp = np.zeros((4+2+B.shape[-1]+2, 4+2+B.shape[-1]+2))
A_temp[[-2, -1], [-2, -1]] = np.array([1, -1])
A.append(A_temp)


# # w11^2 + w22^2 = s^2
# A_temp = np.zeros((4+2+B.shape[-1]+2, 4+2+B.shape[-1]+2))
# A_temp[[0, 3, -2], [0, 3, -2]] = np.array([1, 1, -1])
# A.append(A_temp)
# # w12^2 + w21^2 = s^2
# A_temp = np.zeros((4+2+B.shape[-1]+2, 4+2+B.shape[-1]+2))
# A_temp[[1, 2, -2], [1, 2, -2]] = np.array([1, 1, -1])
# A.append(A_temp)


with np.printoptions(precision=3, suppress=True):
    print(Q)
    print(Q_new)
    print(A0)

X = cp.Variable((4+2+B.shape[-1]+2, 4+2+B.shape[-1]+2), symmetric=True)
constraints = [X >> 0]
constraints += [
    cp.trace(A0 @ X) == 1,
]
constraints += [cp.trace(A_i @ X) == 0 for A_i in A]
prob = cp.Problem(cp.Minimize(cp.trace(Q_new @ X)),
                  constraints)
prob.solve(solver=cp.CLARABEL, max_iter=100)


with np.printoptions(precision=6, suppress=True):
    print(prob.status)
    print("The optimal value is", prob.value)
    print("A solution X is")
    print(X.value)
    print(np.linalg.matrix_rank(X.value))
    X_val = X.value
    X_val = np.delete(X_val, -2, axis=0)
    X_val = np.delete(X_val, -2, axis=1)
    u, s, vt = np.linalg.svd(X_val)
    print(s)
    s[1:] = 0.
    X_val = u @ np.diag(s) @ vt
    print(np.linalg.matrix_rank(X_val))

    # print(sp.linalg.lu(X.value))
    e, v = np.linalg.eig(X.value)
    ind = np.argmax(e)
    sol = v[:, ind] * np.sqrt(e[ind])
    print(sol * np.sign(sol[-1]))

    # A = np.arange(1, 10)[None, :]
    # A = A.T @ A
    # e, v = np.linalg.eig(A)
    # ind = np.argmax(e)
    # print(np.linalg.matrix_rank(A))
    # print(e.shape, v.shape)
    # print(v[:, ind] * np.sqrt(e[ind]))
    print(R_init.T, R_init.T @ t_init)
