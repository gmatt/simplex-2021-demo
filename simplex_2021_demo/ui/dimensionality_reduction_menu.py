from typing import Dict, Literal, NamedTuple, Optional

import streamlit as st

labels = {
    "PCA": "PCA (Principal Component Analysis)",
    "TSNE": "TSNE (t-distributed Stochastic Neighbor Embedding)",
    "UMAP": "UMAP (Uniform Manifold Approximation and Projection)",
}


class DimReductionOption(NamedTuple):
    method: Literal[
        "simple_projection",
        "PCA",
        "TSNE",
        "UMAP",
    ]
    params: Optional[Dict] = None


def select_dim_reduction(
    n_dims: int,
    default: str = "PCA",
    key: int = None,
) -> DimReductionOption:
    default_index = list(labels.keys()).index(default)

    st.write(f"{n_dims = }")
    if n_dims <= 2:
        st.write(
            f"💡 Data is {n_dims}-dimensional so dimensionality reduction panel is empty."
        )
        return DimReductionOption(method="simple_projection")
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
        if method == "project to plane":
            x_axis = int(
                st.number_input(
                    label="x axis dimension",
                    min_value=0,
                    max_value=n_dims - 1,
                    value=0,
                    key=key,
                )
            )
            y_axis = int(
                st.number_input(
                    label="y axis dimension",
                    min_value=0,
                    max_value=n_dims - 1,
                    value=1,
                    key=key,
                )
            )
            return DimReductionOption(
                method="simple_projection",
                params={"x": x_axis, "y": y_axis},
            )
        if method == "reduce dimensions":
            dr_method = st.selectbox(
                label="dimensionality reduction method",
                options=list(labels.keys()),
                index=default_index,
                format_func=lambda x: labels[x],
                key=key,
            )
            return DimReductionOption(method=dr_method)
