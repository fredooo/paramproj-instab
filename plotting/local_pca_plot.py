"""Local PCA ellipse plot for variance visualization."""

import numpy as np
from matplotlib.patches import Ellipse

from plotting.base_2d_plot import Base2DPlot


class LocalPCAPlot(Base2DPlot):
    """PCA ellipses showing local variance structure per cluster."""

    def __init__(
        self,
        Z_clusters,
        anchors,
        xlim=None,
        ylim=None,
        pad=0.1,
        n_std=3.0,
        draw_axes=True,
        draw_bias_line=True,
        draw_points=True,
        point_size=8,
        point_alpha=0.05,
        axis_lw=1.5,
        bias_line_lw=2.5,
        ellipse_lw=2.0,
        ellipse_alpha=1.0,
        min_points=3,
        figsize=(6, 6),
        dpi=150,
        filename="local_pca.png",
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
        self.n_std = float(n_std)
        self.draw_axes = bool(draw_axes)
        self.draw_bias_line = bool(draw_bias_line)
        self.draw_points = bool(draw_points)
        self.point_size = float(point_size)
        self.point_alpha = float(point_alpha)
        self.axis_lw = float(axis_lw)
        self.bias_line_lw = float(bias_line_lw)
        self.ellipse_lw = float(ellipse_lw)
        self.ellipse_alpha = float(ellipse_alpha)
        self.min_points = int(min_points)

    def _cluster_pca(self, Z):
        """Compute PCA for a cluster. Returns (center, eigvals, eigvecs) or None."""
        center = Z.mean(axis=0)

        # If cluster is tiny or degenerate, return None
        if len(Z) < self.min_points:
            return None

        X = Z - center

        # 2x2 covariance (rowvar=False)
        cov = np.cov(X, rowvar=False)
        if cov.shape != (2, 2):
            return None

        # Numerical safety
        if not np.isfinite(cov).all():
            return None

        # Eigen-decomposition (symmetric)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort descending
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Clamp small negatives due to numerical noise
        eigvals = np.maximum(eigvals, 0.0)

        return center, eigvals, eigvecs

    def plot_bias_lines(self):
        """Draw line from anchor to cluster center (visualizes D_bias)."""
        for i, Z in enumerate(self.Z_clusters):
            Z = np.asarray(Z)
            if len(Z) < self.min_points:
                continue
            center = Z.mean(axis=0)
            anchor = self.anchors[i]
            self.ax.plot(
                [anchor[0], center[0]],
                [anchor[1], center[1]],
                color=self.colors[i],
                linewidth=self.bias_line_lw,
                zorder=3,
            )

    def plot_pca_ellipses(self):
        for i, Z in enumerate(self.Z_clusters):
            Z = np.asarray(Z)
            res = self._cluster_pca(Z)
            if res is None:
                continue

            center, eigvals, eigvecs = res

            # If covariance is (near) zero, skip
            if eigvals[0] <= 0.0 and eigvals[1] <= 0.0:
                continue

            # Ellipse axes lengths for n_std "sigma" contour
            # For a Gaussian, sqrt(eigval) is std along that principal direction.
            width = 2.0 * self.n_std * np.sqrt(eigvals[0])  # major axis length
            height = 2.0 * self.n_std * np.sqrt(eigvals[1])  # minor axis length

            # Angle in degrees: eigenvector of largest eigenvalue
            v0 = eigvecs[:, 0]
            angle = np.degrees(np.arctan2(v0[1], v0[0]))

            ell = Ellipse(
                xy=(center[0], center[1]),
                width=width,
                height=height,
                angle=angle,
                fill=False,
                edgecolor=self.colors[i],
                linewidth=self.ellipse_lw,
                alpha=self.ellipse_alpha,
                zorder=2,
            )
            self.ax.add_patch(ell)

            if self.draw_axes:
                # Draw principal axes as line segments centered at the mean
                # Axis half-lengths correspond to n_std * std in each direction
                std0 = self.n_std * np.sqrt(eigvals[0])
                std1 = self.n_std * np.sqrt(eigvals[1])

                # Major axis
                p0a = center - std0 * eigvecs[:, 0]
                p0b = center + std0 * eigvecs[:, 0]
                self.ax.plot(
                    [p0a[0], p0b[0]],
                    [p0a[1], p0b[1]],
                    color=self.colors[i],
                    linewidth=self.axis_lw,
                    zorder=2,
                )

                # Minor axis
                p1a = center - std1 * eigvecs[:, 1]
                p1b = center + std1 * eigvecs[:, 1]
                self.ax.plot(
                    [p1a[0], p1b[0]],
                    [p1a[1], p1b[1]],
                    color=self.colors[i],
                    linewidth=self.axis_lw,
                    zorder=2,
                )

    def render(self):
        if self.draw_points:
            self.plot_points(size=self.point_size, alpha=self.point_alpha)
        self.plot_pca_ellipses()
        if self.draw_bias_line:
            self.plot_bias_lines()
        self.plot_anchors()
        self.finalize()
