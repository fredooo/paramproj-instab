"""Scatter plot visualization for cluster points."""

import numpy as np

from plotting.base_2d_plot import Base2DPlot


class ScatterPlot(Base2DPlot):
    """Basic scatter plot of cluster points with optional anchors."""

    def __init__(
        self,
        Z_clusters,
        anchors=None,
        xlim=None,
        ylim=None,
        pad=0.1,
        point_size=8,
        point_alpha=0.05,
        figsize=(6, 6),
        dpi=150,
        filename="scatter.png",
    ):
        if anchors is None:
            anchors = np.zeros((len(Z_clusters), 2))
            self._draw_anchors = False
        else:
            self._draw_anchors = True

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
        self.point_size = float(point_size)
        self.point_alpha = float(point_alpha)

    def render(self):
        self.plot_points(size=self.point_size, alpha=self.point_alpha)
        if self._draw_anchors:
            self.plot_anchors()
        self.finalize()
