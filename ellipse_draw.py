import numpy as np
from bokeh.layouts import column
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import RdBu11
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar
from bokeh.layouts import gridplot


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
    c = (m2 + n2 - 1.0) / 3.0
    c2 = c**2
    c3 = c * c2
    d = c3 + m2 * n2
    q = d + m2 * n2
    g = m + m * n2

    co = np.zeros_like(px)

    # Case: d < 0 (trigonometric solution)
    neg_mask = d < 0
    if np.any(neg_mask):
        m_, n_, c_, q_, l_, g_ = m[neg_mask], n[neg_mask], c[neg_mask], q[neg_mask], l[neg_mask], g[neg_mask]
        c3_ = c_ ** 3
        acos_arg = np.clip(q_ / c3_, -1.0, 1.0)
        h = np.arccos(acos_arg) / 3.0
        s = np.cos(h) + 2.0
        t = np.sin(h) * np.sqrt(3.0)
        rx = np.sqrt(np.maximum(0.0, m_**2 - c_ * (s + t)))
        ry = np.sqrt(np.maximum(0.0, m_**2 - c_ * (s - t)))
        co[neg_mask] = ry + np.sign(l_) * rx + np.abs(g_) / (rx * ry + eps)

    # Case: d ≥ 0 (algebraic solution)
    pos_mask = ~neg_mask
    if np.any(pos_mask):
        m_, n_, c_, c2_, d_, q_, g_ = m[pos_mask], n[pos_mask], c[pos_mask], c2[pos_mask], d[pos_mask], q[pos_mask], g[pos_mask]
        h = 2.0 * m_ * n_ * np.sqrt(d_)
        s = np.cbrt(np.maximum(1e-8, q_ + h))
        t = c2_ / s
        rx = -(s + t) - 4.0 * c_ + 2.0 * m_**2
        ry = (s - t) * np.sqrt(3.0)
        rm = np.sqrt(rx**2 + ry**2)
        co[pos_mask] = ry / np.sqrt(np.maximum(1e-8, rm - rx)) + 2.0 * g_ / (rm + eps)

    # Final closest point in normalized space
    co = (co - m) / 2.0
    si = np.sqrt(np.maximum(0.0, 1.0 - co**2))
    closest_local = np.stack([a_work * co, b_work * si], axis=1)

    # Signed distance
    dist = np.linalg.norm(p_work - closest_local, axis=1)
    dist *= np.sign(py - closest_local[:, 1])

    # Reverse axis swap
    closest_local[swap] = closest_local[swap, ::-1]
    closest = closest_local * np.sign(p_orig)

    return dist, closest

def ellipse_sdf_iter(p, ab, iters=3):
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
    """Create and display the SDF visualization and error histograms."""
    ellipse_axes = np.array([0.1, 1.0])
    grid_size = 1000  # smaller for faster computation
    plot_range = 5.0
    sample_step = 10
    max_iters = 6

    coords = np.linspace(-plot_range, plot_range, grid_size)
    xx, yy = np.meshgrid(coords, coords)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    # Baseline distances (accurate)
    distances, closest = ellipse_sdf(points, ellipse_axes)

    histograms = []
    bin_edges = np.linspace(0, 1e-3, 10)

    import time
    tic = time.time()

    for i in range(max_iters):
        distances_iter, _ = ellipse_sdf_iter(points, ellipse_axes, iters=i)
        error = np.abs(distances - distances_iter)
        argmax = np.argmax(error)

        # Compute histogram
        hist, edges = np.histogram(error, bins=bin_edges)

        p_hist = figure(height=150, width=400, title=f"Error Histogram @ Iter {i}", y_axis_label="Count",
                        x_axis_label="Error", tools="")
        p_hist.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="navy", line_color="white")
        histograms.append(p_hist)
        print(i, error[argmax], points[argmax])

    print("Average time per iter:", (time.time() - tic) / max_iters)

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

    # Sparse sample points
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

    # Combine main plot and histograms
    layout = column(plot, gridplot([histograms[i:i+3] for i in range(0, len(histograms), 3)]))
    show(layout)

if __name__ == "__main__":
    create_sdf_plot()