import numpy as np

import bokeh.plotting as plt
from bokeh.util.browser import get_browser_controller

n_points = 1000
src_points = np.random.rand(n_points, 2)

theta = np.random.uniform(low=0.0, high=np.pi * 2, size=None)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
t = np.random.rand(2)
tgt_points = src_points @ R.T + t[None, :]



thetas = np.linspace(0, np.pi*2, num=359, endpoint=False)
c_th = np.cos(thetas)
s_th = np.sin(thetas)
Rs = np.block([[[c_th], [-s_th]],
               [[s_th], [c_th]]]).swapaxes(2, 0)

# Bokeh setup
filename = "sdf_circle.html"
plt.output_file(filename, title="SDF Contour of Unit Circle")
fig = plt.figure(width=500, height=500, x_range=(-2, 2), y_range=(-2, 2), match_aspect=True,
           title="Signed Distance Field: Unit Circle")
fig.scatter(c_th, s_th)

# Save and open (for WSL2)
plt.save(fig)
get_browser_controller().open(filename)
