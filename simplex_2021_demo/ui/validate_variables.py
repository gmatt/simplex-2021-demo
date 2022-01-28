from typing import Dict

import numpy as np
import streamlit as st


def validate_variables(
    variables: Dict,
):
    if not ("X" in variables and isinstance(variables["X"], np.ndarray)):
        st.warning("`X` (corpus) should be a numpy array.")
    if not ("y" in variables):
        st.warning("`y` (labels) are not defined. This is fine.")
        variables["y"] = np.ones(len(variables["X"]))
    if not ("input_baseline" in variables):
        st.warning(
            "`input_baseline` is not defined. Initializing to `np.zeros_like(X)`."
        )
        variables["input_baseline"] = np.zeros_like(variables["X"])
    if "model" not in variables and "Model" in variables:
        st.warning("Model() is not initialized. Initializing with `model = Model()`.")
        variables["model"] = variables["Model"]()
    if not hasattr(variables["model"], "latent_representation"):
        st.warning("`model` should have a `latent_representation` method.")
    if "feature_names" not in variables:
        st.warning("`feature_names` are not defined. This is fine.")
        variables["feature_names"] = [f"f{n}" for n in range(variables["X"].shape[1])]
