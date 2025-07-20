import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import RdBu11
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar

def ellipse_sdf(p, ab, iters=8):
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



def create_sdf_plot():
    """Create and display the SDF visualization with closest points."""
    ellipse_axes = np.array([0.1, 1.0])
    grid_size = 1000
    plot_range = 2.0
    sample_step = 10

    coords = np.linspace(-plot_range, plot_range, grid_size)
    xx, yy = np.meshgrid(coords, coords)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    import time
    tic = time.time()
    for _ in range(10):
        distances, closest = ellipse_sdf(points, ellipse_axes)
    print((time.time() - tic) / 10.)
    distance_field = distances.reshape(grid_size, grid_size)

    output_file("ellipse_sdf_visualization.html")

    mapper = LinearColorMapper(palette=RdBu11[::-1], low=-1.0, high=2.0)

    plot = figure(title="Signed Distance Field with Closest Points",
                  width=700, height=700,
                  x_range=(-plot_range, plot_range),
                  y_range=(-plot_range, plot_range),
                  match_aspect=True, tools="pan,wheel_zoom,reset")

    plot.image(image=[distance_field],
               x=-plot_range, y=-plot_range,
               dw=2 * plot_range, dh=2 * plot_range,
               color_mapper=mapper)

    color_bar = ColorBar(color_mapper=mapper, label_standoff=12)
    plot.add_layout(color_bar, 'right')

    # Sparse points
    sample_indices = np.arange(0, len(points), sample_step * grid_size)
    sample_points = points[sample_indices]
    sample_closest = closest[sample_indices]

    source = ColumnDataSource(data=dict(
        x1=sample_points[:, 0],
        y1=sample_points[:, 1],
        x2=sample_closest[:, 0],
        y2=sample_closest[:, 1],
    ))

    plot.circle('x1', 'y1', size=4, color='black', alpha=0.8, legend_label="Query Points", source=source)
    plot.circle('x2', 'y2', size=4, color='green', alpha=0.8, legend_label="Closest Points", source=source)
    plot.segment('x1', 'y1', 'x2', 'y2', color="gray", alpha=0.6, source=source)

    theta = np.linspace(0, 2 * np.pi, 300)
    plot.line(ellipse_axes[0] * np.cos(theta), ellipse_axes[1] * np.sin(theta), color='yellow', line_width=3, legend_label="Ellipse")

    plot.legend.location = "top_left"
    show(plot)


if __name__ == "__main__":
    create_sdf_plot()
