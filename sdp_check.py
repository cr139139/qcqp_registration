import dill
import numpy as np
import cvxpy as cp
import scipy as sp
import matplotlib.pyplot as plt
from solver import SDP

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


X_init = B0
theta_init = np.random.uniform(low=0.0, high=np.pi * 2, size=None)
R_init = np.array([[np.cos(theta_init), -np.sin(theta_init)],
                   [np.sin(theta_init), np.cos(theta_init)]])
t_init = np.random.rand(2)
t_init = np.zeros(2)

X = X_init @ R_init.T + t_init[None, :]

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
X_scatter = ax.scatter(*np.copy(X).T, color='r')
Y_scatter = ax.scatter(*np.copy(X).T, color='b')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

sdp = SDP({"B0": B0, "B": B}, reg=1e-4)


for i in range(1000):
    D, Y = ellipse_sdf(X, np.array([i*0.9/50.+0.1, 1.0]), iters=5)
    X_scatter.set_offsets(np.copy(X))
    Y_scatter.set_offsets(np.copy(Y))
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show(block=False)

    result = sdp.solve(Y)
    R = result["R"]
    t = result["t"]
    c = result["c"]
    print(result["eigenval"] / result["eigenval"][0])

    X = (B0 + (B @ c[None, :, None])[:, :, 0]) @ R.T + t



