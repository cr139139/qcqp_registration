import dill
import numpy as np
import cvxpy as cp
import scipy as sp
import matplotlib.pyplot as plt
from solver import SDP, QCQP_ADMM
import time

np.random.seed(0)

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
    dist = np.sqrt(dx**2 + dy**2)
    dist *= np.where(outside, 1.0, -1.0)

    closest = np.stack((cx, cy), axis=1) * p_sign
    return dist, closest.real


with open('asm_double.pkl', 'rb') as f:
    pca = dill.load(f)
B0 = pca.mean.reshape((-1, 2))  # 359x2
B = pca.components.T
B = B.reshape((-1, 2, B.shape[-1]))  # 359x2xc


thetas = np.linspace(0, np.pi*2, num=359, endpoint=False)
c_th = np.cos(thetas)
s_th = np.sin(thetas)
X_init = np.stack([c_th * np.random.uniform(low=0.1, high=1.0, size=1), s_th]).T
theta_init = np.random.uniform(low=0.0, high=np.pi * 2, size=None)
R_init = np.array([[np.cos(theta_init), -np.sin(theta_init)],
                   [np.sin(theta_init), np.cos(theta_init)]])
t_init = np.random.rand(2)
# t_init = np.zeros(2)

X = X_init @ R_init.T + t_init[None, :]

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
X_scatter = ax.scatter(*np.copy(X).T, color='r')
Y_scatter = ax.scatter(*np.copy(X).T, color='b')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

admm = QCQP_ADMM({"B0": B0, "B": B}, reg=1e-4)
sdp = SDP({"B0": B0, "B": B}, reg=1e-4)

for i in range(1000):
    D, Y = ellipse_sdf(X, np.array([0.1, 1.0]), iters=5)
    # Y = X_init @ R_init.T + t_init[None, :]
    X_scatter.set_offsets(np.copy(X))
    Y_scatter.set_offsets(np.copy(Y))
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show(block=False)

    tic_admm = time.time()
    result = admm.solve(Y, iter=10, rho=1e-0, sigma=0.)
    toc_admm = time.time()
    R = result["R"]
    t = result["t"]
    c = result["c"]
    admm_cost = result["cost"]

    tic_sdp = time.time()
    result = sdp.solve(Y)
    toc_sdp = time.time()
    # R = result["R"]
    # t = result["t"]
    # c = result["c"]
    sdp_cost = result["cost"]
    eigenval = result["eigenval"] / result["eigenval"][0]

    cost_diff = admm_cost - sdp_cost
    print(cost_diff)
    print(f"ADMM time {toc_admm - tic_admm:.5f}, SDP time {toc_sdp - tic_sdp:.5f}, Cost diff: {cost_diff[-1]:.10f}, Optimality: {(cost_diff[1:]<1e-4).all()}")

    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # ax.plot(cost_diff, color='b')
    # # ax.axhline(y=0, color='r', linestyle='--')
    # # ax.set_yscale('log')
    # plt.show()
    # exit()

    result = admm.solve_no_constraint(Y)
    # R = result["R"]
    # t = result["t"]
    # c = result["c"]

    print(R.T @ R)
    print(c)

    X = (B0 + (B @ c[None, :, None])[:, :, 0]) @ R.T + t



