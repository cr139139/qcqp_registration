import numpy as np
import cvxpy as cp

class SDP:
    def __init__(self, ASM, reg=1e-4):
        self.B = ASM["B"]
        self.B0 = ASM["B0"]

        self.Q = np.eye(4 + 2 + self.B.shape[-1])
        self.p = np.zeros(4 + 2 + self.B.shape[-1])
        self.Q_new = np.eye(4 + 2 + self.B.shape[-1] + 2) * 0.

        self.A0 = np.zeros((4 + 2 + self.B.shape[-1] + 2, 4 + 2 + self.B.shape[-1] + 2))
        self.A = []

        self.precompute(reg=reg)

    def precompute(self, reg=1e-4):
        self.Q[6:6 + self.B.shape[-1], 4:6] = self.B.mean(axis=0).T
        self.Q[4:6, 6:6 + self.B.shape[-1]] = self.B.mean(axis=0)
        BB = self.B[:, :, :, None] * self.B[:, :, None, :]
        self.Q[6:6 + self.B.shape[-1], 6:6 + self.B.shape[-1]] = BB.mean(axis=0).sum(axis=0)
        self.Q[6:6 + self.B.shape[-1], 6:6 + self.B.shape[-1]] += np.eye(self.B.shape[-1]) * reg

        self.p[4:6] = 2 * self.B0.mean(axis=0)
        self.p[6:6 +self. B.shape[-1]] = 2 * (self.B0[:, :, None] * self.B).sum(axis=1).mean(axis=0)

        # rot+trans+asm+scale+slack
        self.Q_new[-1, -1] = (self.B0 ** 2).sum(axis=1).mean()

        # lambda^2 = 1.
        self.A0[-1, -1] = 1.
        # w11^2 + w12^2 = s^2
        A_temp = np.zeros((4 + 2 + self.B.shape[-1] + 2, 4 + 2 + self.B.shape[-1] + 2))
        A_temp[[0, 1, -2], [0, 1, -2]] = np.array([1, 1, -1])
        self.A.append(A_temp)
        # w21^2 + w22^2 = s^2
        A_temp = np.zeros((4 + 2 + self.B.shape[-1] + 2, 4 + 2 + self.B.shape[-1] + 2))
        A_temp[[2, 3, -2], [2, 3, -2]] = np.array([1, 1, -1])
        self.A.append(A_temp)
        # w11 * w21 + w12 * w22 = 0
        A_temp = np.zeros((4 + 2 + self.B.shape[-1] + 2, 4 + 2 + self.B.shape[-1] + 2))
        A_temp[[0, 2, 1, 3], [2, 0, 3, 1]] = np.array([1, 0, 1, 0])
        self.A.append(A_temp)
        # w11 * w22 - w12 * w21 = s^2
        A_temp = np.zeros((4 + 2 + self.B.shape[-1] + 2, 4 + 2 + self.B.shape[-1] + 2))
        A_temp[[0, 1, 3, 2, -2], [3, 2, 0, 1, -2]] = np.array([1, -1, 1, -1, -2])
        self.A.append(A_temp)
        # s^2 = w^2
        A_temp = np.zeros((4 + 2 + self.B.shape[-1] + 2, 4 + 2 + self.B.shape[-1] + 2))
        A_temp[[-2, -1], [-2, -1]] = np.array([1, -1])
        self.A.append(A_temp)


    def solve(self, Y):
        YY = Y[:, :, None] * Y[:, None, :]
        YY_mean = YY.mean(axis=0)
        self.Q[0:2, 0:2] = self.Q[2:4, 2:4] = YY_mean
        self.Q[4:5, 0:2] = self.Q[5:6, 2:4] = -Y.mean(axis=0, keepdims=True)
        self.Q[0:2, 4:5] = self.Q[2:4, 5:6] = -Y.mean(axis=0, keepdims=True).T
        BY = (self.B[:, :, None, :] * Y[:, None, :, None]).reshape((-1, 4, self.B.shape[-1]))
        self.Q[6:6 + self.B.shape[-1], 0:4] = -BY.mean(axis=0).T
        self.Q[0:4, 6:6 + self.B.shape[-1]] = -BY.mean(axis=0)

        B0Y = (self.B0[:, :, None] * Y[:, None, :]).reshape((-1, 4))
        self.p[0:4] = -2 * B0Y.mean(axis=0)

        # rot+trans+asm+scale+slack
        self.Q_new[0:4 + 2 + self.B.shape[-1], 0:4 + 2 + self.B.shape[-1]] = self.Q
        self.Q_new[4 + 2 + self.B.shape[-1] + 1:4 + 2 + self.B.shape[-1] + 2, 0:4 + 2 + self.B.shape[-1]] = self.p / 2.
        self.Q_new[0:4 + 2 + self.B.shape[-1], 4 + 2 + self.B.shape[-1] + 1:4 + 2 + self.B.shape[-1] + 2] = self.p[None].T / 2.

        Z = cp.Variable((4 + 2 + self.B.shape[-1] + 2, 4 + 2 + self.B.shape[-1] + 2), symmetric=True)
        constraints = [Z >> 0]
        constraints += [cp.trace(self.A0 @ Z) == 1]
        constraints += [cp.trace(A_i @ Z) == 0 for A_i in self.A]
        prob = cp.Problem(cp.Minimize(cp.trace(self.Q_new @ Z)), constraints)
        prob.solve(solver=cp.CLARABEL)
        Z = Z.value

        Z_temp = np.delete(Z, -2, axis=0)
        Z_temp = np.delete(Z_temp, -2, axis=1)
        e, v = np.linalg.eig(Z_temp)
        ind = np.argmax(e)
        sol = v[:, ind] * np.sqrt(e[ind])
        sol = sol * np.sign(sol[-1])

        R = sol[0:4].reshape((2, 2))
        s = np.sqrt(Z[-2, -2])
        t = sol[4:6]
        c = sol[6:-1]

        R_real = R.T
        s_real = 1 / s
        t_real = s_real * (R_real @ t)

        z = sol[0:-1]
        cost = z.T @ self.Q @ z + self.p @ z + (self.B0 ** 2).sum(axis=1).mean()

        result = {"R": R_real, "t": t_real, "c": c, "s": s, "eigenval": e, "cost": cost}

        return result


class QCQP_ADMM:
    def __init__(self, ASM, reg=1e-4):
        self.B = ASM["B"]
        self.B0 = ASM["B0"]

        self.Q = np.eye(4 + 2 + self.B.shape[-1])
        self.p = np.zeros(4 + 2 + self.B.shape[-1])
        self.G = np.zeros((4, 4 + 2 + self.B.shape[-1]))
        self.G[0:4, 0:4] = np.eye(4)

        self.z = np.zeros(4 + 2 + self.B.shape[-1])
        self.z[0:4] = np.eye(2).reshape(-1)
        self.xi = np.zeros(4)
        self.nu = np.zeros(4)

        self.precompute(reg=reg)

    def precompute(self, reg=1e-4):
        self.Q[6:6 + self.B.shape[-1], 4:6] = self.B.mean(axis=0).T
        self.Q[4:6, 6:6 + self.B.shape[-1]] = self.B.mean(axis=0)
        BB = self.B[:, :, :, None] * self.B[:, :, None, :]
        self.Q[6:6 + self.B.shape[-1], 6:6 + self.B.shape[-1]] = BB.mean(axis=0).sum(axis=0)
        self.Q[6:6 + self.B.shape[-1], 6:6 + self.B.shape[-1]] += np.eye(self.B.shape[-1]) * reg

        self.p[4:6] = 2 * self.B0.mean(axis=0)
        self.p[6:6 +self. B.shape[-1]] = 2 * (self.B0[:, :, None] * self.B).sum(axis=1).mean(axis=0)

    def step(self, Y, rho=1e-3, sigma=1e-3):
        YY = Y[:, :, None] * Y[:, None, :]
        YY_mean = YY.mean(axis=0)
        self.Q[0:2, 0:2] = self.Q[2:4, 2:4] = YY_mean
        self.Q[4:5, 0:2] = self.Q[5:6, 2:4] = -Y.mean(axis=0, keepdims=True)
        self.Q[0:2, 4:5] = self.Q[2:4, 5:6] = -Y.mean(axis=0, keepdims=True).T
        BY = (self.B[:, :, None, :] * Y[:, None, :, None]).reshape((-1, 4, self.B.shape[-1]))
        self.Q[6:6 + self.B.shape[-1], 0:4] = -BY.mean(axis=0).T
        self.Q[0:4, 6:6 + self.B.shape[-1]] = -BY.mean(axis=0)

        B0Y = (self.B0[:, :, None] * Y[:, None, :]).reshape((-1, 4))
        self.p[0:4] = -2 * B0Y.mean(axis=0)
        self.z = np.linalg.solve(self.Q*2 + sigma*np.eye(self.z.shape[0]) + self.G.T @ self.G * rho,
                                 -self.p + sigma*self.z + self.G.T @ (rho * self.xi - self.nu))
        xi_temp = self.G @ self.z + self.nu / rho
        U, _, Vt = np.linalg.svd(xi_temp.reshape((2, 2)))
        det = np.linalg.det(U @ Vt)
        Vt[1, 0:2] *= det
        self.xi = (U @ Vt).flatten()
        self.nu = self.nu + rho * (self.G @ self.z - self.xi)

    def solve(self, Y, iter=10, rho=1e-3, sigma=1e-3):
        costs = []
        for i in range(iter):
            self.step(Y, rho, sigma)
            z_i = np.copy(self.z)
            U, _, Vt = np.linalg.svd(z_i[0:4].reshape((2, 2)))
            det = np.linalg.det(U @ Vt)
            Vt[1, 0:2] *= det
            z_i[0:4] = (U @ Vt).flatten()
            cost = z_i.T @ self.Q @ z_i + self.p @ z_i + (self.B0 ** 2).sum(axis=1).mean()
            costs.append(cost)

        R = z_i[0:4].reshape((2, 2))
        t = z_i[4:6]
        s = 1.
        c = z_i[6:]

        R_real = R.T
        s_real = 1 / s
        t_real = s_real * (R_real @ t)
        result = {"R": R_real, "t": t_real, "c": c, "s": s, "cost": costs}

        return result

    def solve_no_constraint(self, Y):
        self.step(Y)
        z = np.linalg.solve(self.Q*2, -self.p)
        R = z[0:4].reshape((2, 2))
        t = z[4:6]
        c = z[6:]

        cost = z.T @ self.Q @ z + self.p @ z + (self.B0 ** 2).sum(axis=1).mean()
        R_real = np.linalg.inv(R)

        result = {"R": R_real, "t": R_real @ t, "c": c, "cost": cost}

        return result


