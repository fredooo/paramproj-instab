"""KDE contour plot for density visualization."""
import numpy as np
from scipy.stats import gaussian_kde

from plotting.base_2d_plot import Base2DPlot


class KDEContourPlot(Base2DPlot):
    """Kernel density estimation contours for each cluster."""

    def __init__(
        self,
        Z_clusters,
        anchors,
        xlim=None,
        ylim=None,
        pad=0.1,
        levels=None,
        gridsize=300,
        contour_lw=0.8,
        figsize=(6, 6),
        dpi=150,
        filename="kde_contours.png",
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
        self.levels = levels if levels is not None else (0.60, 0.75, 0.85, 0.90, 0.95, 0.99)
        self.gridsize = gridsize
        self.contour_lw = contour_lw

    def plot_contours(self):
        """Draw KDE density contours at specified probability levels."""
        xx, yy = np.mgrid[
            self.xlim[0] : self.xlim[1] : self.gridsize * 1j,
            self.ylim[0] : self.ylim[1] : self.gridsize * 1j,
        ]
        grid = np.vstack([xx.ravel(), yy.ravel()])

        cell_area = (
            (self.xlim[1] - self.xlim[0])
            * (self.ylim[1] - self.ylim[0])
            / grid.shape[1]
        )

        for i, Z in enumerate(self.Z_clusters):
            if len(Z) < 10:
                continue

            kde = gaussian_kde(Z.T)
            zz = kde(grid).reshape(xx.shape)

            z_sorted = np.sort(zz.ravel())[::-1]
            cumsum = np.cumsum(z_sorted * cell_area)
            cumsum /= cumsum[-1]

            thresholds = []
            for lvl in self.levels:
                idx = np.searchsorted(cumsum, lvl)
                if idx < len(z_sorted):
                    thresholds.append(z_sorted[idx])

            # Ensure strictly increasing and unique levels
            thresholds = np.unique(np.sort(thresholds))

            # Need at least 2 distinct levels for contour
            if len(thresholds) < 2:
                continue

            self.ax.contour(
                xx,
                yy,
                zz,
                levels=thresholds,
                colors=[self.colors[i]],
                linewidths=self.contour_lw,
                zorder=2,
            )


    def render(self):
        self.plot_contours()
        self.plot_anchors()
        self.finalize()
