import numpy as np
import streamlit as st
from Simplex.models.base import BlackBox

from simplex_2021_demo.util import numpy_to_torch, torch_to_numpy


@st.cache()
def get_latents(model: BlackBox, x: np.ndarray) -> np.ndarray:
    x = numpy_to_torch(x)
    x = model.latent_representation(x)
    x = torch_to_numpy(x)
    return x
