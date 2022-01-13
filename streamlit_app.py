import importlib
import sys
from typing import Any, Callable, List, Tuple, TypedDict, Union

import numpy as np
import openTSNE
import pandas as pd
import plotly.graph_objects as go
import sklearn.decomposition
import streamlit as st
import streamlit.components.v1 as components
import torch
import umap
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots
from pyprojroot import here
from streamlit_ace import st_ace
from streamlit_plotly_events import plotly_events

sys.path.append(str(here("Simplex")))

from explainers.simplex import Simplex
from models.base import BlackBox
import simplex_2021_demo.temp_code


def freeze_numpy(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


class Result(TypedDict):
    result: List[Tuple[float, np.ndarray, np.ndarray]]
    n_keep: int
    corpus_examples: np.ndarray
    corpus_latents: np.ndarray
    corpus_labels: np.ndarray
    selected_index: int
    sort_id: np.ndarray
    input_baseline: np.ndarray
    model: BlackBox


@st.cache
def perform_analysis(
    code,
    n_keep=5,
    selected_index: int = 0,
) -> Result:
    here("simplex_2021_demo/temp_code.py").write_text(code)
    importlib.reload(simplex_2021_demo.temp_code)

    model = simplex_2021_demo.temp_code.model
    X = simplex_2021_demo.temp_code.X
    y = simplex_2021_demo.temp_code.y

    assert isinstance(X, np.ndarray), "X should be a numpy array."

    if selected_index >= len(X):
        st.session_state.selected_index = 0
        selected_index = 0

    i = selected_index

    corpus_latents = (
        model.latent_representation(torch.from_numpy(X.astype(np.float32)))
        .detach()
        .numpy()
    )

    test_inputs = X

    test_latents = (
        model.latent_representation(torch.from_numpy(test_inputs.astype(np.float32)))
        .detach()
        .numpy()
    )

    simplex = Simplex(
        corpus_examples=torch.from_numpy(X.astype(np.float32)),
        corpus_latent_reps=torch.from_numpy(corpus_latents.astype(np.float32)),
    )

    simplex.fit(
        test_examples=torch.from_numpy(test_inputs.astype(np.float32)),
        test_latent_reps=torch.from_numpy(test_latents.astype(np.float32)),
        n_keep=n_keep,
    )

    input_baseline = torch.zeros(
        X.shape
    )  # Baseline tensor of the same shape as corpus_inputs
    simplex.jacobian_projection(test_id=i, model=model, input_baseline=input_baseline)

    result, sort_id = simplex.decompose(i, return_id=True)

    return {
        "result": [
            (
                w_c,
                freeze_numpy(x_c.detach().numpy()),
                freeze_numpy(proj_jacobian_c.detach().numpy()),
            )
            for w_c, x_c, proj_jacobian_c in result
        ],
        "n_keep": n_keep,
        "corpus_examples": freeze_numpy(X),
        "corpus_latents": freeze_numpy(corpus_latents),
        "corpus_labels": freeze_numpy(y),
        "selected_index": selected_index,
        "sort_id": freeze_numpy(sort_id),
        "input_baseline": freeze_numpy(input_baseline.detach().numpy()),
        "model": model,
    }


def visualize_instance(instance: Any, index: int):
    st.write("c" + str(index))
    st.write(instance)


def plot_jacobian_table(
    values: np.ndarray,
    saliencies: np.ndarray,
    column_names: Union[List, None] = None,
    i=0,
):
    if column_names is None:
        column_names = [f"feature {n}" for n in range(len(values))]
    st.latex(f"x^{{{i}}}")

    df = pd.DataFrame(
        data=np.vstack([values, saliencies]).T,
        columns=["value", "saliency"],
        index=column_names,
    )

    # Same colormap as in the paper.
    cmap = LinearSegmentedColormap.from_list("rg", ["r", "w", "g"], N=256)
    st.dataframe(
        df.style.background_gradient(
            axis=0,
            gmap=saliencies,
            cmap=cmap,
            vmin=-0.25,
            vmax=0.25,
        ).format("{:.2f}")
    )


@st.cache
def PCA(X):
    pca = sklearn.decomposition.PCA()
    pca.fit(X)
    return pca.transform


@st.cache
def TSNE(X):
    tsne = openTSNE.TSNE()
    tsne = tsne.fit(X)
    return tsne.transform


@st.cache
def UMAP(X):
    umap_ = umap.UMAP()
    umap_.fit(X)
    return umap_.transform


def get_or_train_dim_reduction(
    X: np.array, key: int
) -> Callable[[np.ndarray], np.ndarray]:
    n_dims = X.shape[1]
    st.write(f"{n_dims = }")
    if n_dims <= 2:
        st.write(
            f":bulb: Data is {n_dims}-dimensional so dimensionality reduction panel is empty."
        )
        return lambda x: x
    else:
        method = st.selectbox(
            label="mode",
            options=[
                "project to plane",
                "reduce dimensions",
            ],
            index=1,
            key=key,
        )
        # TODO Add 3d scatter plots.
        if method == "project to plane":
            x_axis = int(
                st.number_input(
                    "x axis dimension",
                    0,
                    n_dims - 1,
                    0,
                    key=key,
                )
            )
            y_axis = int(
                st.number_input(
                    "y axis dimension",
                    0,
                    n_dims - 1,
                    1,
                    key=key,
                )
            )
            return lambda x: x[:, [x_axis, y_axis]]
        if method == "reduce dimensions":
            # TODO Add full names.
            dr_method = st.selectbox(
                label="dimensionality reduction method",
                options=[
                    "PCA",
                    "TSNE",
                    "UMAP",
                ],
                # TODO Ugly, different defaults for input and latent.
                index=1 if key != 0 else 0,
                key=key,
            )
            if dr_method == "PCA":
                return PCA(X)
            elif dr_method == "TSNE":
                return TSNE(X)
            elif dr_method == "UMAP":
                return UMAP(X)


def plot_figures(result: Result):

    selected_index = result["selected_index"]
    explanation_indices = list(result["sort_id"][: result["n_keep"]])

    with st.expander("visualization settings"):
        col1, col2 = st.columns(2)
        with col1:
            input_dim_reducer = get_or_train_dim_reduction(
                result["corpus_examples"], key=0
            )
            corpus_examples = input_dim_reducer(result["corpus_examples"])
        with col2:
            latent_dim_reducer = get_or_train_dim_reduction(
                result["corpus_latents"], key=1
            )
            corpus_latents = latent_dim_reducer(result["corpus_latents"])

    t = np.arange(0.0, 1.0, 0.01)

    line1 = [
        np.array(
            [
                (
                    input_dim_reducer(
                        tt
                        * result["input_baseline"][
                            explanation_index : explanation_index + 1
                        ]
                        + (1 - tt)
                        * result["corpus_examples"][
                            explanation_index : explanation_index + 1
                        ]
                    )
                )[0]
                for tt in t
            ]
        )
        for explanation_index in explanation_indices
    ]
    # TODO Temp remove.
    if "UMAP" not in str(latent_dim_reducer):
        line2 = [
            np.array(
                [
                    latent_dim_reducer(
                        result["model"]
                        .latent_representation(
                            torch.from_numpy(
                                tt
                                * result["input_baseline"][
                                    explanation_indices[i : i + 1]
                                ]
                                + (1 - tt)
                                * result["corpus_examples"][
                                    explanation_indices[i : i + 1]
                                ]
                            )
                        )
                        .detach()
                        .numpy()
                    )[0]
                    for tt in t
                ]
            )
            for i in range(len(explanation_indices))
        ]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Input space",
            "Latent space",
        ),
    )

    fig.add_trace(
        go.Scatter(
            name="input_points",
            x=corpus_examples[:, 0],
            y=corpus_examples[:, 1],
            text=list(range(len(result["corpus_examples"]))),
            textposition="top right",
            mode="markers+text",
            marker=dict(
                color=[
                    "cyan"
                    if i == selected_index
                    else "green"
                    if i in explanation_indices
                    else y
                    for i, y in enumerate(result["corpus_labels"])
                ],
                colorscale="Bluered",
                symbol="x",
            ),
        ),
        row=1,
        col=1,
    )

    # TODO Make projection dimensions configurable.
    fig.add_trace(
        go.Scatter(
            name="latent_points",
            x=corpus_latents[:, 0],
            y=corpus_latents[:, 1],
            text=list(range(len(result["corpus_examples"]))),
            textposition="top right",
            mode="markers+text",
            marker=dict(
                color=[
                    "cyan"
                    if i == selected_index
                    else "green"
                    if i in explanation_indices
                    else y
                    for i, y in enumerate(result["corpus_labels"])
                ],
                colorscale="Bluered",
                symbol="x",
            ),
        ),
        row=1,
        col=2,
    )

    if "UMAP" not in str(input_dim_reducer):
        for l in line1:
            fig.add_trace(
                go.Scatter(
                    x=l[:, 0],
                    y=l[:, 1],
                    mode="lines",
                    marker=dict(
                        color="rgba(0, 0, 0, 1)",
                    ),
                ),
                row=1,
                col=1,
            )

    # TODO Temp remove.
    if "UMAP" not in str(latent_dim_reducer):
        for l in line2:
            fig.add_trace(
                go.Scatter(
                    x=l[:, 0],
                    y=l[:, 1],
                    mode="lines",
                    marker=dict(
                        color="rgba(0, 0, 0, 1)",
                    ),
                ),
                row=1,
                col=2,
            )

    fig.update_layout(showlegend=False)

    # The javascript code uses indices instead of names (I couldn't get names to work).
    # Make sure that we don't break the order.
    assert [x.name for x in fig.select_traces()][:2] == [
        "input_points",
        "latent_points",
    ], "Plot traces are in wrong order."

    # fig.update_layout(height=600)

    st.write("Click *twice* to select point for explanation.")
    selected_points = plotly_events(fig)

    # st.plotly_chart(fig, use_container_width=True)

    # language=js
    javascript = """const script = window.parent.document.createElement("script");
script.src = "https://cdn.plot.ly/plotly-2.8.3.min.js";
window.parent.document.head.append(script);

setInterval(() => {
    let x = window.parent.document.querySelector("div.js-plotly-plot");

    if (x == null) {
        x = window.parent.document.querySelector('iframe[title="streamlit_plotly_events.plotly_events"]')
            .contentDocument
            .querySelector("div.js-plotly-plot")
    }


    if (x != null && x.DONE == null) {
        x.on("plotly_hover", e => {
            var pointIndex = e.points[0].pointNumber;

            window.parent.Plotly.Fx.hover(x, [
                {curveNumber: 0, pointNumber: pointIndex},
                {curveNumber: 1, pointNumber: pointIndex},
            ], ["xy", "x2y2"]);
        });
        x.DONE = true;
    }
}, 1000);
"""

    components.html(f"<script>\n{javascript}\n</script>", height=0)

    if selected_points:
        st.session_state.selected_index = selected_points[0]["pointIndex"]


def visualize_result(result: Result):
    plot_figures(result)

    cols = st.columns([3, 1] + [2, 3] * result["n_keep"])
    with cols[0]:
        plot_jacobian_table(
            values=result["result"][selected_index][1],
            saliencies=np.zeros_like(result["result"][selected_index][1]),
            i=selected_index,
        )
    with cols[1]:
        st.markdown("#")
        st.markdown("#")
        st.latex("=")
    for i in range(result["n_keep"]):
        with cols[i * 2 + 2]:
            st.markdown("#")
            st.markdown("#")
            st.latex(rf"{result['result'][i][0]*100 :.2f}\%\;\times")
        with cols[i * 2 + 3]:
            index = result["sort_id"][i]
            # visualize_instance(result["corpus_examples"][index], index)
            plot_jacobian_table(
                values=result["result"][i][1],
                saliencies=result["result"][i][2],
                i=index,
            )


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    example_models = sorted(
        x.stem for x in here("simplex_2021_demo/example_models").glob("*.py")
    )
    example_model = st.sidebar.selectbox(
        "example models",
        example_models,
        index=example_models.index("simple_projection"),
    )
    code_default = (
        here("simplex_2021_demo/example_models") / f"{example_model}.py"
    ).read_text()

    st.write("edit code:")
    code = st_ace(
        value=code_default,
        language="python",
        auto_update=True,
        font_size=12,
    )

    n_keep = int(st.number_input("n_keep", 1, None, 3, 1))

    if "selected_index" not in st.session_state:
        st.session_state.selected_index = 0

    selected_index = st.session_state.selected_index

    result = perform_analysis(
        code=code,
        n_keep=n_keep,
        selected_index=selected_index,
    )

    visualize_result(result)
