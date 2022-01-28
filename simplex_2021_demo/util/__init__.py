import numpy as np
import streamlit as st
import torch


def numpy_to_torch(x: np.ndarray) -> torch.tensor:
    return torch.from_numpy(x.astype(np.float32))


def torch_to_numpy(x: torch.tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def padding(size: int = 1):
    """Vertical alignment in Streamlit, an elaborate solution."""
    for _ in range(size):
        st.markdown("#")
