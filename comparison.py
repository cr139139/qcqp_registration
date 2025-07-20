import dill
import numpy as np
import matplotlib.pyplot as plt
from solver import SDP, QCQP_ADMM
import time
from tqdm import tqdm


with open('asm_single.pkl', 'rb') as f:
    pca = dill.load(f)
B0 = pca.mean.reshape((-1, 2))  # 359x2
B = pca.components.T
B = B.reshape((-1, 2, B.shape[-1]))  # 359x2xc

admm = QCQP_ADMM({"B0": B0, "B": B}, reg=1e-4)
sdp = SDP({"B0": B0, "B": B}, reg=1e-4)

tests = []
admm_times = []
sdp_times = []

for i in range(100):
    thetas = np.linspace(0, np.pi * 2, num=359, endpoint=False)
    c_th = np.cos(thetas)
    s_th = np.sin(thetas)
    X_init = np.stack([c_th * np.random.uniform(low=0.1, high=1.0, size=1), s_th]).T
    theta_init = np.random.uniform(low=0.0, high=np.pi * 2, size=None)
    R_init = np.array([[np.cos(theta_init), -np.sin(theta_init)],
                       [np.sin(theta_init), np.cos(theta_init)]])
    t_init = np.random.rand(2)
    Y = X_init @ R_init.T + t_init[None, :]

    tic_admm = time.time()
    result = admm.solve(Y, iter=20, rho=1e-0, sigma=0.)
    toc_admm = time.time()
    R = result["R"]
    t = result["t"]
    c = result["c"]
    admm_cost = result["cost"]

    tic_sdp = time.time()
    result = sdp.solve(Y)
    toc_sdp = time.time()
    R = result["R"]
    t = result["t"]
    c = result["c"]
    sdp_cost = result["cost"]
    eigenval = result["eigenval"] / result["eigenval"][0]

    cost_diff = admm_cost - sdp_cost
    tests.append(cost_diff)
    admm_times.append(toc_admm-tic_admm)
    sdp_times.append(toc_sdp-tic_sdp)


tests = np.array(tests)

mean = tests.mean(axis=0)
std = np.std(tests, axis=0)
iters = np.arange(1, len(mean) + 1)

plt.figure(figsize=(8, 5))
plt.plot(iters, mean, label='Mean Error', color='navy', linewidth=2)
plt.fill_between(iters, mean - std, mean + std, color='skyblue', alpha=0.3, label='±1 Std Dev')

plt.xlabel("ADMM Iterations", fontsize=12)
plt.ylabel("Error vs SDP", fontsize=12)
plt.title(f"ADMM vs SDP Error\nADMM: {np.mean(admm_times):.3f}s, SDP: {np.mean(sdp_times):.3f}s, Shape dim={B.shape[-1]}", fontsize=13)
plt.xlim(1, len(mean))
plt.ylim(tests.min(), tests.max())
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()