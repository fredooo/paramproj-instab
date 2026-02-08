"""Convenience function to render all plot types."""

from plotting.anchor_line_plot import AnchorLinePlot
from plotting.kde_contour_plot import KDEContourPlot
from plotting.local_pca_plot import LocalPCAPlot
from plotting.scatter_plot import ScatterPlot
from plotting.voronoi_plot import VoronoiPlot


def plot_all(Z_clusters, anchors, filename_prefix):
    """Render all visualization types for a given set of clusters and anchors."""
    KDEContourPlot(
        Z_clusters=Z_clusters,
        anchors=anchors,
        filename=f"{filename_prefix}_kde_contours.png",
    ).render()

    LocalPCAPlot(
        Z_clusters=Z_clusters,
        anchors=anchors,
        n_std=3.0,
        draw_axes=True,
        filename=f"{filename_prefix}_local_pca.png",
    ).render()

    AnchorLinePlot(
        Z_clusters=Z_clusters,
        anchors=anchors,
        show_points=True,
        point_alpha=0.4,
        line_lw=0.5,
        filename=f"{filename_prefix}_anchor_lines.png",
    ).render()

    VoronoiPlot(
        Z_clusters=Z_clusters,
        anchors=anchors,
        filename=f"{filename_prefix}_voronoi.png",
    ).render()

    ScatterPlot(
        Z_clusters=Z_clusters,
        anchors=anchors,
        filename=f"{filename_prefix}_scatter.png",
    ).render()
