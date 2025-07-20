import numpy as np
import bokeh.plotting as plt
from bokeh.util.browser import get_browser_controller


class PCA:
    def __init__(self, n):
        self.n = n
        self.mean = self.components = self.explained_variance_ratio = None

    def fit(self, X):
        Xc = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.mean = X.mean(axis=0)
        self.components = Vt[:self.n]
        self.explained_variance_ratio = (S[:self.n]**2) / (S**2).sum()

    def transform(self, X):
        return (X - self.mean) @ self.components.T

    def inverse(self, Z):
        return Z @ self.components + self.mean




thetas = np.linspace(0, np.pi*2, num=359, endpoint=False)
c_th = np.cos(thetas)
s_th = np.sin(thetas)
shapes = [np.stack([c_th, s_th]).T]
for i in range(9):
    shapes.append(np.stack([c_th * 0.1 * (i + 1), s_th]).T)
    # shapes.append(np.stack([c_th, s_th * 0.1 * (i + 1)]).T)
shapes = np.stack(shapes)  # 10x359x2

pca = PCA(n=1)
X = shapes.reshape((shapes.shape[0], -1))
pca.fit(X)
Z = pca.transform(X)
X_ = pca.inverse(Z)
print(pca.explained_variance_ratio.sum())

import dill
with open('asm_single.pkl', 'wb') as f:
    # pickle.dump(MyClass, f)
    dill.dump(pca, f, fmode=dill.CONTENTS_FMODE)


# filename = "sdf_circle.html"
# plt.output_file(filename, title="SDF Contour of Unit Circle")
# fig = plt.figure(width=500, height=500, x_range=(-2, 2), y_range=(-2, 2), match_aspect=True,
#            title="Signed Distance Field: Unit Circle")
# # for i in range(10):
# #     fig.scatter(*shapes[i].T)
# fig.scatter(*(X_[0]-X[0]).reshape((-1, 2)).T)
#
# # Save and open (for WSL2)
# plt.save(fig)
# get_browser_controller().open(filename)



