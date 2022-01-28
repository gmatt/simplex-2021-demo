from typing import Callable

import numpy as np
import openTSNE
import sklearn
import streamlit as st
import umap

from simplex_2021_demo.ui.dimensionality_reduction_menu import DimReductionOption


@st.cache()
def PCA(X):
    pca = sklearn.decomposition.PCA()
    pca.fit(X)
    return pca.transform


@st.cache()
def TSNE(X):
    tsne = openTSNE.TSNE()
    tsne = tsne.fit(X)
    return tsne.transform


@st.cache()
def UMAP(X):
    umap_ = umap.UMAP()
    umap_.fit(X)
    return umap_.transform


def fit_dim_reducer(
    option: DimReductionOption,
    X: np.array,
) -> Callable[[np.ndarray], np.ndarray]:
    n_dims = X.shape[1]
    method, params = option

    if method == "simple_projection":
        if n_dims <= 2:
            return lambda x: x
        return lambda x: x[:, [params["x"], params["y"]]]
    elif method == "PCA":
        return PCA(X)
    elif method == "TSNE":
        return TSNE(X)
    elif method == "UMAP":
        return UMAP(X)
