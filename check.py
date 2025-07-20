import numpy as np
from bokeh.plotting import figure, output_file, save
from bokeh.palettes import Sunset11, interp_palette
from bokeh.util.browser import get_browser_controller
from bokeh.layouts import row, column

# Parameters
center = (0.0, 0.0)
radius = 1.0
margin = 4.0  # Extra space around the shape
res = 200     # Grid resolution (per axis)
palette = interp_palette(Sunset11, 9)

# Derived values
xlim = (center[0] - radius - margin, center[0] + radius + margin)
ylim = (center[1] - radius - margin, center[1] + radius + margin)
x, y = np.meshgrid(np.linspace(*xlim, res), np.linspace(*ylim, res))
sdf = np.hypot(x - center[0], y - center[1]) - radius
levels = np.linspace(-1, 7, len(palette))

# Bokeh setup
filename = "sdf_circle.html"
output_file(filename, title="SDF Contour of Unit Circle")
p = figure(width=500, height=500, x_range=xlim, y_range=ylim, match_aspect=True,
           title="Signed Distance Field: Unit Circle")
contour = p.contour(x, y, sdf, levels=levels, fill_color=palette, line_color="black")
# p.add_layout(contour.construct_color_bar(), "right")

dummy = figure(height=500,
               width=120,
               min_border=0,)
dummy.add_layout(contour.construct_color_bar(), 'right')

# Save and open (for WSL2)
save(row(p, dummy))
get_browser_controller().open(filename)
