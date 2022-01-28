from dataclasses import dataclass
from typing import List, NamedTuple

import numpy as np
import streamlit as st
from Simplex.explaiers.simplex import Simplex
from Simplex.models.base import BlackBox

from simplex_2021_demo.util import numpy_to_torch, torch_to_numpy


@dataclass
class Explanation:
    weight: float
    value: np.ndarray
    proj_jacobian: np.ndarray


class SimplexResult(NamedTuple):
    result: List[Explanation]
    sort_id: np.ndarray


@st.cache()
def simplex(
    model: BlackBox,
    corpus_examples: np.ndarray,
    corpus_latents: np.ndarray,
    test_inputs: np.ndarray,
    test_latents: np.ndarray,
    input_baseline: np.ndarray,
    n_keep: int,
    test_id: int,
) -> SimplexResult:
    """
    Interface around SimplEx.

    Args:
        model:
        corpus_examples:
        corpus_latents:
        test_inputs:
        test_latents:
        input_baseline:
        n_keep:
        test_id:

    Returns:

    """
    simplex = Simplex(
        corpus_examples=numpy_to_torch(corpus_examples),
        corpus_latent_reps=numpy_to_torch(corpus_latents),
    )

    simplex.fit(
        test_examples=numpy_to_torch(test_inputs),
        test_latent_reps=numpy_to_torch(test_latents),
        n_keep=n_keep,
    )

    simplex.jacobian_projection(
        test_id=test_id,
        model=model,
        input_baseline=numpy_to_torch(input_baseline),
    )
    result, sort_id = simplex.decompose(test_id=test_id, return_id=True)

    return SimplexResult(
        result=[
            Explanation(
                weight=w_c,
                value=torch_to_numpy(x_c),
                proj_jacobian=torch_to_numpy(proj_jacobian_c),
            )
            for w_c, x_c, proj_jacobian_c in result
        ],
        sort_id=sort_id,
    )
