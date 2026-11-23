"""Plotting tools for an initialized GUNDAM interface."""

from __future__ import annotations

import re
from math import isfinite
from operator import index
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .interface import GundamInterface
    from .internal.dial import DialView
    from .internal.parameters import ParameterView


class DrawingTools:
    """Create diagnostic plots from a GUNDAM interface."""

    def __init__(self, interface: GundamInterface) -> None:
        self._interface = interface

    def drawDialResponseDensity(
        self,
        parameterName: str,
        *,
        parameterSetIndex: int | None = None,
        sigmaRange: tuple[float, float] = (-4.0, 4.0),
        nPoints: int = 200,
        nDials: int = 1000,
        nBins: int = 350,
        dpi: float = 150,
        showKnots: bool = True,
        figsize: tuple[float, float] = (8, 5),
    ) -> tuple[Figure, Axes]:
        """Plot the percentage of sampled dials crossing each bin.

        Select an enabled parameter by exact name, optionally within the
        zero-based parameter set index. Scan ``nPoints`` intervals within
        ``sigmaRange`` around the prior, clipped to the parameter bounds. Use
        the parameter bounds directly when the standard deviation is undefined.
        Select up to ``nDials`` dials evenly across the event caches and use
        ``nBins`` bins along each axis.

        The interface must be initialized. Return the Matplotlib figure and
        main axes for display or export by the caller.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            from matplotlib.lines import Line2D
        except ModuleNotFoundError as error:
            if error.name != "matplotlib":
                raise
            raise ImportError("Install gundam-interface[plot] to use DrawingTools") from error

        self._interface._requireInitialized()
        nPoints, nDials, nBins = index(nPoints), index(nDials), index(nBins)
        if min(nPoints, nDials, nBins) < 1:
            raise ValueError("nPoints, nDials, and nBins must be positive")
        sigmaMin, sigmaMax = sigmaRange
        if not (isfinite(sigmaMin) and isfinite(sigmaMax) and sigmaMin < sigmaMax):
            raise ValueError("sigmaRange must contain two finite, increasing values")

        model = self._interface.getModel()
        parameterSets = model.getParametersManager().getParameterSetList()
        if parameterSetIndex is not None:
            if not 0 <= parameterSetIndex < len(parameterSets):
                raise ValueError(f"parameterSetIndex must be in [0, {len(parameterSets)})")
            parameterSets = [parameterSets[parameterSetIndex]]
        matches = [
            parameter
            for parameterSet in parameterSets
            for parameter in parameterSet.getParameterList()
            if parameter.isEnabled() and parameter.getName() == parameterName
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one enabled parameter named {parameterName!r}, found {len(matches)}"
            )
        parameter = matches[0]
        allDials = [
            dial
            for cache in model.getDialCacheList()
            for dial in cache.getDialListAffecting(parameter)
        ]
        if not allDials:
            raise ValueError(f"No dials found for {parameterName!r}")
        nSampled = min(nDials, len(allDials))
        dials = [allDials[i * len(allDials) // nSampled] for i in range(nSampled)]
        prior, stdDev = parameter.getPrior(), parameter.getStdDev()
        useSigma = isfinite(stdDev) and stdDev > 0
        lower, upper = _getParameterBounds(parameter)
        if useSigma:
            lower = max(lower, prior + sigmaMin * stdDev)
            upper = min(upper, prior + sigmaMax * stdDev)
        if not (isfinite(lower) and isfinite(upper) and lower < upper):
            raise ValueError(f"No finite scan interval for {parameterName!r}")

        values = np.linspace(lower, upper, nPoints + 1)
        knots = set()
        if showKnots:
            knots = {knot for dial in dials for knot in dial.getSourceKnotPositions()}
        responses = _scanResponses(parameter, dials, values)
        xEdges, yEdges, density = _computeDensity(values, responses, nBins)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_facecolor("black")
        cmap = plt.get_cmap("magma").with_extremes(bad="black")
        mesh = ax.pcolormesh(
            xEdges, yEdges, np.ma.masked_less_equal(density.T, 0),
            norm=LogNorm(vmin=10.0 / nSampled, vmax=100), cmap=cmap,
        )
        fig.colorbar(mesh, ax=ax, extend="min").set_label("Fraction of dials passing through [%]")
        ax.axhline(1.0, color="white", linestyle="--", linewidth=0.8)
        priorLine = ax.axvline(prior, color="white", linestyle="--", linewidth=0.8, label="prior")
        legendHandles = [priorLine]
        visibleKnots = sorted(knot for knot in knots if lower <= knot <= upper)
        for knot in visibleKnots:
            ax.axvline(knot, color="skyblue", linestyle=":", linewidth=0.9, zorder=4)
        if visibleKnots:
            legendHandles.insert(0, Line2D([], [], color="skyblue", linestyle=":", label="knots"))
        ax.set(xlim=(lower, upper), ylim=(yEdges[0], yEdges[-1]))
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("Dial response")
        if useSigma:
            sigmaAxis = ax.secondary_xaxis("bottom", functions=(
                lambda value: (value - prior) / stdDev,
                lambda sigma: prior + sigma * stdDev,
            ))
            sigmaAxis.set_xlabel("Distance from prior [sigma]")
            sigmaAxis.spines["bottom"].set_position(("outward", 42))
        ax.set_title(f"Density of {nSampled} GUNDAM dial responses\n({parameterName})")
        ax.grid(alpha=0.3)
        ax.legend(
            handles=legendHandles, facecolor="black", edgecolor="gray",
            labelcolor="white", framealpha=0.85,
        )
        fig.tight_layout()
        return fig, ax


def _getParameterBounds(parameter: ParameterView) -> tuple[float, float]:
    # The native Python binding exposes the parameter bounds in the summary.
    bounds = re.search(r"bounds=\[([^,]+),\s*([^\]]+)\]", parameter.getSummary())
    if bounds is None:
        raise ValueError(f"Cannot read bounds for {parameter.getName()!r}")
    lower, upper = map(float, bounds.groups())
    return (lower if isfinite(lower) else -np.inf, upper if isfinite(upper) else np.inf)


def _scanResponses(
    parameter: ParameterView, dials: list[DialView], values: np.ndarray,
) -> np.ndarray:
    originalValue = parameter.getValue()
    responses = np.empty((len(values), len(dials)))
    try:
        for i, value in enumerate(values):
            parameter.setValue(float(value))
            for j, dial in enumerate(dials):
                responses[i, j] = dial.evaluateResponse()
    finally:
        parameter.setValue(originalValue)
        for dial in dials:
            dial.updateInputBuffer()
    if not np.isfinite(responses).all():
        raise ValueError(f"Non-finite dial responses for {parameter.getName()!r}")
    return responses


def _computeDensity(
    values: np.ndarray, responses: np.ndarray, nBins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xEdges = np.linspace(values[0], values[-1], nBins + 1)
    yMin, yMax = responses.min(), responses.max()
    if yMin == yMax:
        padding = max(abs(yMin) * 0.01, 0.01)
        yMin, yMax = yMin - padding, yMax + padding
    yEdges = np.linspace(yMin, yMax, nBins + 1)

    # Within each x bin, a piecewise linear curve crosses all y bins between
    # its minimum and maximum. These extrema are at scan points or bin edges.
    segments = np.searchsorted(values, xEdges, side="right") - 1
    segments = np.clip(segments, 0, len(values) - 2)
    fractions = (xEdges - values[segments]) / (values[segments + 1] - values[segments])
    edgeResponses = responses[segments] + fractions[:, None] * (
        responses[segments + 1] - responses[segments]
    )
    minimum = np.minimum(edgeResponses[:-1], edgeResponses[1:])
    maximum = np.maximum(edgeResponses[:-1], edgeResponses[1:])
    xBins = np.clip(np.searchsorted(xEdges, values, side="right") - 1, 0, nBins - 1)
    np.minimum.at(minimum, xBins, responses)
    np.maximum.at(maximum, xBins, responses)

    first = np.clip(np.searchsorted(yEdges, minimum, side="right") - 1, 0, nBins - 1)
    last = np.clip(np.searchsorted(yEdges, maximum, side="right") - 1, 0, nBins - 1)
    # Mark each occupied y interval once per dial and recover counts by summing.
    changes = np.zeros((nBins, nBins + 1), dtype=int)
    columns = np.arange(nBins)[:, None]
    np.add.at(changes, (columns, first), 1)
    np.add.at(changes, (columns, last + 1), -1)
    counts = changes[:, :-1].cumsum(axis=1)
    return xEdges, yEdges, 100.0 * counts / responses.shape[1]
