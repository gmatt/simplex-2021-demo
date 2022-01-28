from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from simplex_2021_demo.main_logic.simplex import Explanation
from simplex_2021_demo.util import padding

# Same colormap as in the paper.
cmap = LinearSegmentedColormap.from_list("rg", ["r", "w", "g"], N=256)


def plot_jacobian_table(
    values: np.ndarray,
    saliencies: np.ndarray,
    feature_names: Optional[List] = None,
    i=0,
):
    if feature_names is None:
        feature_names = [f"f{n}" for n in range(len(values))]
    st.latex(f"x^{{{i}}}")

    df = pd.DataFrame(
        data=np.vstack([values, saliencies]).T,
        columns=["value", "saliency"],
        index=feature_names,
    )

    st.dataframe(
        df.style.background_gradient(
            axis=0,
            gmap=saliencies,
            cmap=cmap,
            vmin=-0.25,
            vmax=0.25,
        ).format("{:.2f}")
    )


def visualize_decomposition(
    result: List[Explanation],
    explanation_indices: List[int],
    n_keep: int,
    selected_index: int,
    feature_names: List[str],
):
    # Horizontal layout with percentages and `jacobian_table`s, similar to the paper.

    col1, col2, *cols = st.columns([3, 1] + [2, 3] * n_keep)
    with col1:
        plot_jacobian_table(
            values=result[selected_index].value,
            saliencies=np.zeros_like(result[selected_index].value),
            i=selected_index,
            feature_names=feature_names,
        )
    with col2:
        padding(2)
        st.latex("=")
    for i in range(n_keep):
        with cols[i * 2]:
            padding(2)
            st.latex(rf"{result[i].weight * 100 :.2f}\%\;\times")
        with cols[i * 2 + 1]:
            index = explanation_indices[i]
            plot_jacobian_table(
                values=result[i].value,
                saliencies=result[i].proj_jacobian,
                i=index,
                feature_names=feature_names,
            )
