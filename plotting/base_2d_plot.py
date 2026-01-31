"""Base class for 2D projection plots."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


class Base2DPlot:
    """Base class providing common functionality for 2D cluster visualization."""

    def __init__(
        self,
        Z_clusters,
        anchors,
        xlim=None,
        ylim=None,
        pad=0.1,
        figsize=(6, 6),
        dpi=150,
        filename="plot.png",
    ):
        self.Z_clusters = [np.asarray(Z) for Z in Z_clusters]
        self.anchors = np.asarray(anchors)

        if len(self.Z_clusters) != len(self.anchors):
            raise ValueError("Number of clusters and number of anchors must match.")

        self.pad = pad
        self.filename = filename
        self.dpi = dpi

        self.fig, self.ax = plt.subplots(figsize=figsize)

        cmap = get_cmap("tab10")
        self.colors = [cmap(i % 10) for i in range(len(self.Z_clusters))]

        self.xlim, self.ylim = self._compute_limits(xlim, ylim)

    def _compute_limits(self, xlim, ylim):
        if xlim is not None and ylim is not None:
            return xlim, ylim

        # Stack only REAL data for span computation
        all_real = np.vstack(self.Z_clusters + [self.anchors])

        xmin, ymin = all_real.min(axis=0)
        xmax, ymax = all_real.max(axis=0)

        # Square extent
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)

        width = xmax - xmin
        height = ymax - ymin
        span = max(width, height)

        # Apply padding in data space
        half_size = 0.5 * span * (1.0 + 2.0 * self.pad)

        # Final limits
        xlim = (cx - half_size, cx + half_size)
        ylim = (cy - half_size, cy + half_size)

        return xlim, ylim


    def plot_points(self, size=8, alpha=0.05):
        """Draw scatter points for each cluster."""
        for i, Z in enumerate(self.Z_clusters):
            if len(Z) == 0:
                continue
            self.ax.scatter(
                Z[:, 0], Z[:, 1],
                c=[self.colors[i]],
                s=size,
                alpha=alpha,
                linewidths=0.0,
                zorder=1,
            )

    def plot_anchors(self, marker="X", size=120, edgecolor="black", linewidth=2.0, zorder=10):
        """Draw anchor points with distinctive markers."""
        for i, anchor in enumerate(self.anchors):
            self.ax.scatter(
                anchor[0], anchor[1],
                c=[self.colors[i]],
                marker=marker,
                s=size,
                edgecolors=edgecolor,
                linewidths=linewidth,
                zorder=zorder,
            )

    def finalize(self):
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)

        self.ax.axis("off")
        self.ax.set_position([0.0, 0.0, 1.0, 1.0])

        self.fig.savefig(self.filename, dpi=self.dpi)
        plt.close(self.fig)
