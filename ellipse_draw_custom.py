import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import RdBu11
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar


def ellipse_sdf(points, ab, eps=1e-8):
    """
    Compute signed distance and closest points on an ellipse (circle if a ≈ b).

    Args:
        points: (n, 2) array of 2D points
        a, b: ellipse semi-axes
        eps: threshold to treat as a circle

    Returns:
        dist: (n,) array of signed distances
        closest: (n, 2) array of closest points on the ellipse
    """
    a, b = ab
    points = np.asarray(points)
    if abs(a - b) < eps:
        # Circle case
        norm = np.linalg.norm(points, axis=1, keepdims=True)
        closest = (points / np.maximum(norm, eps)) * a
        dist = norm[:, 0] - a
        return dist, closest

    p_orig = points
    p = np.abs(points)
    swap = p[:, 0] > p[:, 1]
    p_work = p.copy()
    p_work[swap] = p[swap, ::-1]

    a_work = np.full(len(p), a)
    b_work = np.full(len(p), b)
    a_work[swap], b_work[swap] = b, a

    px, py = p_work[:, 0], p_work[:, 1]

    l = b_work**2 - a_work**2
    m = a_work * px / l
    m2 = m**2
    n = b_work * py / l
    n2 = n**2
    A = (m2 + n2 - 1.0) / 3.0
    B = 2*m2*n2

    co = np.cbrt(-(B+A**3)+np.sqrt(A**6+(B+A**3)**2)) + np.cbrt(-(B+A**3)-np.sqrt(A**6+(B+A**3)**2))
    si = np.sqrt(np.maximum(0.0, 1.0 - co**2))
    closest_local = np.stack([a_work * co, b_work * si], axis=1)

    # Signed distance
    dist = np.linalg.norm(p_work - closest_local, axis=1)
    dist *= np.sign(py - closest_local[:, 1])

    # Reverse axis swap
    closest_local[swap] = closest_local[swap, ::-1]
    closest = closest_local * np.sign(p_orig)

    return dist, closest


def create_sdf_plot():
    """Create and display the SDF visualization with closest points."""
    ellipse_axes = np.array([1.0, 0.6])
    grid_size = 200
    plot_range = 2.0
    sample_step = 10

    coords = np.linspace(-plot_range, plot_range, grid_size)
    xx, yy = np.meshgrid(coords, coords)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    import time
    tic = time.time()
    for _ in range(10):
        distances, closest = ellipse_sdf(points, ellipse_axes)
    print((time.time() - tic)/10.)
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
