"""Anchor line plot showing displacement from anchor to perturbed points."""
import matplotlib.pyplot as plt
import numpy as np

from plotting.base_2d_plot import Base2DPlot

class AnchorLinePlot(Base2DPlot):
    """Lines from perturbed points to their corresponding anchor."""

    def __init__(
        self,
        Z_clusters,
        anchors,
        xlim=None,
        ylim=None,
        pad=0.1,
        show_points=False,
        point_size=8,
        point_alpha=1.0,
        max_lines=100,
        line_lw=0.8,
        figsize=(6, 6),
        dpi=150,
        filename="anchor_lines.png",
    ):
        super().__init__(
            Z_clusters=Z_clusters,
            anchors=anchors,
            xlim=xlim,
            ylim=ylim,
            pad=pad,
            figsize=figsize,
            dpi=dpi,
            filename=filename,
        )
        self.show_points = bool(show_points)
        self.point_size = float(point_size)
        self.point_alpha = float(point_alpha)
        self.max_lines = max_lines
        self.line_lw = float(line_lw)

    def plot_anchor_lines(self):
        """Draw lines from sampled cluster points to their anchor."""
        rng = np.random.default_rng()

        for i, (Z, anchor) in enumerate(zip(self.Z_clusters, self.anchors)):
            Z = np.asarray(Z)
            n = len(Z)
            if n == 0:
                continue

            if self.max_lines is not None and n > self.max_lines:
                idx = rng.choice(n, size=self.max_lines, replace=False)
                Z_sel = Z[idx]
            else:
                Z_sel = Z

            if self.show_points:
                self.ax.scatter(
                    Z_sel[:, 0], Z_sel[:, 1],
                    c=[self.colors[i]],
                    s=self.point_size,
                    alpha=self.point_alpha,
                    linewidths=0.0,
                    zorder=1,
                )

            for z in Z_sel:
                self.ax.plot(
                    [z[0], anchor[0]], [z[1], anchor[1]],
                    color=self.colors[i],
                    linewidth=self.line_lw,
                    zorder=2,
                )

    def plot_mean_displacement_circles(self):
        """Draw circle at anchor with radius = mean displacement."""
        for i, (Z, anchor) in enumerate(zip(self.Z_clusters, self.anchors)):
            Z = np.asarray(Z)
            if len(Z) == 0:
                continue
            dists = np.linalg.norm(Z - anchor, axis=1)
            mean_disp = np.mean(dists)
            circle = plt.Circle(
                anchor, mean_disp,
                fill=False,
                color=self.colors[i],
                alpha=0.3,
                linewidth=1.0,
                zorder=3,
            )
            self.ax.add_patch(circle)

    def render(self):
        self.plot_anchor_lines()
        self.plot_mean_displacement_circles()
        self.plot_anchors()
        self.finalize()
