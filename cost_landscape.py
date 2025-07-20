import dill
import numpy as np
import cvxpy as cp
import scipy as sp
import matplotlib.pyplot as plt
from solver import SDP, QCQP_ADMM
import time

np.random.seed(42)

with open('asm_double.pkl', 'rb') as f:
    pca = dill.load(f)
B0 = pca.mean.reshape((-1, 2))  # 359x2
B = pca.components.T
B = B.reshape((-1, 2, B.shape[-1]))  # 359x2xc
ellipse_a = 1.00


thetas = np.linspace(0, np.pi*2, num=359, endpoint=False)
c_th = np.cos(thetas)
s_th = np.sin(thetas)
# X_init = np.stack([c_th * np.random.uniform(low=0.1, high=1.0, size=1), s_th]).T
X_init = np.stack([c_th * ellipse_a, s_th]).T
theta_init = 0 # np.random.uniform(low=-np.pi, high=np.pi, size=None)
R_init = np.array([[np.cos(theta_init), -np.sin(theta_init)],
                   [np.sin(theta_init), np.cos(theta_init)]])
t_init = np.zeros(2)

admm = QCQP_ADMM({"B0": B0, "B": B}, reg=1e-4)
sdp = SDP({"B0": B0, "B": B}, reg=1e-4)

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

def get_cost(R, t, c):
    X = (B0 + (B @ c[None, :, None])[:, :, 0]) @ R.T + t
    _, Y = ellipse_sdf(X, np.array([ellipse_a, 1.]), iters=100)
    return ((X-Y)**2).sum(axis=0).mean(), X

for i in range(1000):
    Y = X_init @ R_init.T + t_init[None, :]
    _, Y_clone = ellipse_sdf(Y, np.array([1. * ellipse_a, 1.]), iters=10)
    print(((Y_clone-Y)**2).sum(axis=0).mean())

    tic_admm = time.time()
    result = admm.solve(Y, iter=10, rho=1e-0, sigma=0.)
    toc_admm = time.time()
    R_admm = result["R"]
    t_admm = result["t"]
    c_admm = result["c"]
    admm_cost = result["cost"]

    tic_sdp = time.time()
    result = sdp.solve(Y)
    toc_sdp = time.time()
    R_sdp = result["R"]
    t_sdp = result["t"]
    c_sdp = result["c"]
    sdp_cost = result["cost"]
    eigenval = result["eigenval"] / result["eigenval"][0]

    cost_diff = admm_cost - sdp_cost
    print(f"ADMM time {toc_admm - tic_admm:.5f}, SDP time {toc_sdp - tic_sdp:.5f}, Cost diff: {cost_diff[-1]:.10f}, Optimality: {(cost_diff[1:]<1e-4).all()}")

    a_admm = np.arctan2(R_admm[1,0], R_admm[0,0])
    a_sdp = np.arctan2(R_sdp[1,0], R_sdp[0,0])
    thetas = np.linspace(-np.pi / 36., np.pi / 36., num=101, endpoint=True)

    costs_admm = []
    for theta in (thetas+a_admm):
        R_theta = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
        cost_admm, X_temp = get_cost(R_theta, t_admm, c_admm)
        costs_admm.append(cost_admm)
    costs_admm = np.array(costs_admm)

    costs_sdp = []
    for theta in (thetas + a_sdp):
        R_theta = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
        cost_sdp, X_temp = get_cost(R_theta, t_admm, c_admm)
        costs_sdp.append(cost_sdp)
    costs_sdp = np.array(costs_sdp)

    result = admm.solve_no_constraint(Y)
    cost_no_constraint = result["cost"]

    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Common styling params
    line_kwargs = dict(linewidth=2)
    vline_kwargs = dict(color='red', linestyle='--', label='Ground Truth')

    # ADMM Plot
    thetas_admm = np.rad2deg(thetas + a_admm)
    ax[0].plot(thetas_admm, costs_admm, label="ADMM", color='navy', **line_kwargs)
    ax[0].axvline(x=0, **vline_kwargs)
    # ax[0].axhline(y=cost_no_constraint, color='k', linestyle='--', label="No Constraint")
    ax[0].set_title("ADMM Cost Landscape", fontsize=14)
    ax[0].set_xlabel("Rotation Angle (°)", fontsize=12)
    ax[0].set_ylabel("Registration Cost", fontsize=12)
    ax[0].grid(True, linestyle=':', alpha=0.6)
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax[0].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax[0].legend()
    ax[0].set_xlim(thetas_admm.min(), thetas_admm.max())

    # SDP Plot
    thetas_sdp = np.rad2deg(thetas + a_sdp)
    ax[1].plot(thetas_sdp, costs_sdp, label="SDP", color='darkgreen', **line_kwargs)
    ax[1].axvline(x=0, **vline_kwargs)
    # ax[1].axhline(y=cost_no_constraint, color='k', linestyle='--', label="No Constraint")
    ax[1].set_title("SDP Cost Landscape", fontsize=14)
    ax[1].set_xlabel("Rotation Angle (°)", fontsize=12)
    ax[1].grid(True, linestyle=':', alpha=0.6)
    ax[1].xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax[1].legend()
    ax[1].set_xlim(thetas_sdp.min(), thetas_sdp.max())

    # Super title
    fig.suptitle(f"Cost Landscape Comparison for Ellipse ($a$ = {ellipse_a:.2f}, $b$ = 1.00)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


    # cost_sdp, X_temp = get_cost(R_admm, t_admm, c_admm)
    # plt.plot(*X_temp.T)
    # plt.plot(*ellipse_sdf(X_temp, np.array([ellipse_a, 1.]), iters=100)[1].T)
    # plt.axis('equal')
    # plt.show()


    exit()



