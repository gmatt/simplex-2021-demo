from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events

from simplex_2021_demo.util.execute_javascript import execute_javascript


def plot_charts(
    corpus_examples: np.ndarray,
    corpus_latents: np.ndarray,
    n_keep: int,
    selected_index: int,
    explanation_indices: List[int],
    corpus_labels: np.ndarray,
    paths: List[np.ndarray],
    path_latents: List[np.ndarray],
) -> Optional[int]:
    """

    Args:
        corpus_examples:
        corpus_latents:
        n_keep:
        selected_index:
        explanation_indices:
        corpus_labels:
        paths:
        path_latents:

    Returns:
        Index of clicked node, or `None`.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Input space",
            "Latent space",
        ),
    )

    marker_colors = [
        # Cyan if selected.
        "cyan" if i == selected_index else
        # Green if explanation.
        "green" if i in explanation_indices else
        # Else use the colorscale with the label value.
        y
        for i, y in enumerate(corpus_labels)
    ]

    # Points in input and latent space.

    fig.add_trace(
        go.Scatter(
            name="input_points",
            x=corpus_examples[:, 0],
            y=corpus_examples[:, 1],
            text=list(range(len(corpus_examples))),
            textposition="top right",
            mode="markers+text",
            marker=dict(
                color=marker_colors,
                colorscale="Bluered",
                symbol="x",
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            name="latent_points",
            x=corpus_latents[:, 0],
            y=corpus_latents[:, 1],
            text=list(range(len(corpus_examples))),
            textposition="top right",
            mode="markers+text",
            marker=dict(
                color=marker_colors,
                colorscale="Bluered",
                symbol="x",
            ),
        ),
        row=1,
        col=2,
    )

    # Integrated Jacobian paths.

    for path in paths:
        fig.add_trace(
            go.Scatter(
                name="interpolation",
                x=path[:, 0],
                y=path[:, 1],
                mode="lines",
                marker=dict(
                    color="rgba(0, 0, 0, 0.5)",
                ),
            ),
            row=1,
            col=1,
        )

    for path in path_latents:
        fig.add_trace(
            go.Scatter(
                name="interpolation",
                x=path[:, 0],
                y=path[:, 1],
                mode="lines",
                marker=dict(
                    color="rgba(0, 0, 0, 0.5)",
                ),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(showlegend=False)

    # Enable interactive hover groups.
    execute_javascript(
        # language=js
        """
const nKeep = %(n_keep)d;

if (!("Plotly" in window.parent)) {
    const script = document.createElement("script");
    script.src = "https://cdn.plot.ly/plotly-2.8.3.min.js";
    document.head.append(script);
}

const interval = setInterval(() => {
    const plot = document.querySelector('iframe[title="streamlit_plotly_events.plotly_events"]')
        .contentDocument
        .querySelector("div.js-plotly-plot")

    if (plot != null) {
        plot.on("plotly_hover", event => {
            const {pointNumber, curveNumber} = event.points[0];

            if ([0, 1].includes(curveNumber)) {
                window.parent.Plotly.Fx.hover(plot, [
                    {curveNumber: 0, pointNumber: pointNumber},
                    {curveNumber: 1, pointNumber: pointNumber},
                ], ["xy", "x2y2"]);
            } else {
                const otherCurve = curveNumber < nKeep + 2
                    ? curveNumber + nKeep
                    : curveNumber - nKeep;
                window.parent.Plotly.Fx.hover(plot, [
                    {curveNumber: curveNumber, pointNumber: pointNumber},
                    {curveNumber: otherCurve, pointNumber: pointNumber},
                ], ["xy", "x2y2"]);
            }
        });
        clearInterval(interval);
    }
}, 1000);
"""
        % {"n_keep": n_keep}
    )

    selection = plotly_events(
        plot_fig=fig,
        click_event=True,
        override_height="600px",
    )

    if selection:
        return selection[0]["pointIndex"]
