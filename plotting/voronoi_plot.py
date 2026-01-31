"""Voronoi tessellation plot with cluster points."""
import numpy as np

from plotting.base_2d_plot import Base2DPlot


class VoronoiPlot(Base2DPlot):
    """Voronoi regions colored by nearest anchor, with cluster points overlaid."""

    def __init__(
        self,
        Z_clusters,
        anchors,
        xlim=None,
        ylim=None,
        pad=0.1,
        voronoi_alpha=0.6,
        point_size=8,
        point_alpha=1.0,
        figsize=(6, 6),
        dpi=150,
        filename="voronoi.png",
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
        self.voronoi_alpha = float(voronoi_alpha)
        self.point_size = float(point_size)
        self.point_alpha = float(point_alpha)

    def plot_voronoi(self, gridsize=400):
        """Render Voronoi regions as a colored background image."""
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim

        xx, yy = np.meshgrid(
            np.linspace(xmin, xmax, gridsize),
            np.linspace(ymin, ymax, gridsize),
        )
        grid = np.stack([xx, yy], axis=-1)

        dists = np.sum((grid[..., None, :] - self.anchors[None, None, :, :]) ** 2, axis=-1)
        labels = np.argmin(dists, axis=-1)

        color_img = np.zeros((gridsize, gridsize, 4))
        for i, color in enumerate(self.colors):
            mask = labels == i
            color_img[mask, :3] = color[:3]
            color_img[mask, 3] = self.voronoi_alpha

        self.ax.imshow(
            color_img,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            interpolation="nearest",
            zorder=0,
        )

    def render(self):
        self.plot_voronoi()
        self.plot_points(size=self.point_size, alpha=self.point_alpha)
        self.plot_anchors()
        self.finalize()
