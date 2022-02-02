from typing import Dict

import numpy as np
import streamlit as st
from pyprojroot import here

from simplex_2021_demo.main_logic.dimensionality_reduction import fit_dim_reducer
from simplex_2021_demo.main_logic.get_latent_representation import get_latents
from simplex_2021_demo.main_logic.simplex import simplex
from simplex_2021_demo.ui.code_input import code_input
from simplex_2021_demo.ui.dimensionality_reduction_menu import select_dim_reduction
from simplex_2021_demo.ui.validate_variables import validate_variables
from simplex_2021_demo.util.execute_python import run_code_and_get_variables
from simplex_2021_demo.util.shift_indices_after_skipping_one import (
    shift_indices_after_skipping_one,
)
from simplex_2021_demo.util.update_streamlit_state import update_streamlit_state
from simplex_2021_demo.visualization.visualize_decomposition import (
    visualize_decomposition,
)
from simplex_2021_demo.visualization.visualize_geometry import plot_charts


@st.cache(allow_output_mutation=True)
def train_model(code: str) -> Dict:
    return run_code_and_get_variables(code)


def get_corpus(code: str, local_variables: Dict) -> Dict:
    return run_code_and_get_variables(code, local_variables)


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    if "i" not in st.session_state:
        st.session_state.i = 0

    example_models_dir = here("simplex_2021_demo/example_models")
    example_models = sorted(x.name for x in example_models_dir.iterdir() if x.is_dir())
    example_model = st.sidebar.selectbox(
        label="example models",
        options=example_models,
        index=example_models.index("simple_projection"),
    )
    default_model_code = (example_models_dir / example_model / "model.py").read_text()
    default_data_code = (example_models_dir / example_model / "data.py").read_text()

    with st.expander("ℹ️ Info", expanded=True):
        # language=md
        """
        SimplEx is an explainable AI (XAI) method that explains predictions with similar examples (e.g. this example is
        80% like this and 20% like that corpus example) and also provides individual feature importances for each
        example.

        See the paper: [📄 Explaining Latent Representations with a Corpus of Examples](https://arxiv.org/abs/2110.15355)

        This web app lets you experiment with the method with toy examples or real models.

        Choose an example from the sidebar. (If the sidebar is not visible, you can summon it with the arrow in the top
        left corner.)
        Click on the chart or enter a corpus index to create an explanation for a specific example.
        You should see percentages at the bottom, and 3 examples (the explanation) highlighted on the chart. The chart
        shows the same examples in input space and latent space (the latter being the output of the penultimate layer of
        the model).
        Hover should be in sync between the input and latent space charts. The black lines are the paths of the
        integrated Jacobians (along which the feature contributions are calculated). Different dimensionality reduction
        methods (to visualize in 2D) can be chosen in "visualization settings".

        **Using a custom model**

        Enter the code for the model. Arbitrary Python code can be entered into the text field, and the code runs
        immediately. Currently, the app works with PyTorch models, which should be differentiable (they usually are) and
        have a `latent_representation()` method. (This is described in the
        [BlackBox interface](https://github.com/JonathanCrabbe/Simplex/blob/787f01a83783835137819110a309b46dc66418db/models/base.py).)
        Basically, in the simplest case, take any existing PyTorch model, including training code, copy the `forward()`
        method into a new `latent_representation()` method, wherein remove the last step (the last layer) before the
        return.
        You can provide the corpus (which is used to draw explanations from) to the second text field below the first
        one. This can be the training set, or can be different. The variables you need to define are `model`, an
        instance with BlackBox interface as described above, and a corpus `X`, a Numpy array.
        You can optionally define the labels `y`, a Numpy array, `feature_names`, list of strings, and `input_baseline`,
        a Numpy array.
        Streamlit caches states, and it won’t perform the same calculation twice, it just uses the cached results. If
        you’re getting unexpected results, it might be due to a caching issue, press `C` to clear cache and `R` to
        reload the app.

        GitHub: https://github.com/gmatt/simplex-2021-demo

        There seems to be an issue with the code editor, it doesn't load in some browsers. If so, please toggle the
        following checkbox:"""
        use_ace = st.checkbox("use Ace code editor", value=True)
        "Close this info with the `➖` in the top right corner of this box."

    st.write("Enter model code:")
    st.caption(
        "Feel free to enter training code, it will only run once (unless things go wrong, then blame it on Streamlit)."
    )
    model_code = code_input(value=default_model_code, use_ace_editor=use_ace)
    variables = train_model(code=model_code)

    st.write("Enter data (corpus):")
    st.caption(
        "Variables `X` (corpus), `model`, and probably `y` (labels) should exist. You can use variables from the "
        "previous block."
    )
    data_code = code_input(value=default_data_code, use_ace_editor=use_ace)
    variables = get_corpus(code=data_code, local_variables=variables)

    validate_variables(
        variables=variables,
    )

    model = variables["model"]
    corpus_examples = variables["X"]
    corpus_labels = variables["y"]
    input_baseline = variables["input_baseline"]
    corpus_size = corpus_examples.shape[0]
    n_features = corpus_examples.shape[1]
    feature_names = variables["feature_names"]

    col1, col2 = st.columns(2)
    with col1:
        n_keep = int(
            st.number_input(
                label="number of examples (n_keep)",
                min_value=1,
                max_value=corpus_size - 1,
                value=3,
                step=1,
            )
        )
    with col2:
        i = int(
            st.number_input(
                label="selected index",
                min_value=0,
                max_value=corpus_size,
                step=1,
                key="i",
            )
        )

    corpus_latents = get_latents(model, corpus_examples)
    dim_latent = corpus_latents.shape[1]

    if i >= corpus_size:
        st.session_state.i = 0
        i = st.session_state.i

    # We construct explanation for the ith example, so we leave it out from the corpus.
    # `test_inputs` is always a single element, so `test_id` is always 0.
    # np.delete is an immutable operation, it returns a new array.
    result, sort_id = simplex(
        model=model,
        corpus_examples=np.delete(corpus_examples, i, axis=0),
        corpus_latents=np.delete(corpus_latents, i, axis=0),
        test_inputs=corpus_examples[i][np.newaxis],
        test_latents=corpus_latents[i][np.newaxis],
        input_baseline=np.delete(input_baseline, i, axis=0),
        n_keep=n_keep,
        test_id=0,
    )

    explanation_indices = list(sort_id[:n_keep])
    explanation_indices = shift_indices_after_skipping_one(explanation_indices, i)

    # Interpolate from baseline to examples for visualization.
    t = np.arange(0.0, 1.0, 0.01)
    paths = [
        np.vstack(
            [
                np.interp(t, [0, 1], [input_baseline[j][k], corpus_examples[j][k]])
                for k in range(n_features)
            ]
        ).T
        for j in explanation_indices
    ]
    path_latents = [get_latents(model, path) for path in paths]

    # Dimensionality reduction.
    with st.expander("visualization settings"):
        col1, col2 = st.columns(2)
        with col1:
            input_dim_reduction = select_dim_reduction(
                n_dims=n_features, default="PCA", key=0
            )
        with col2:
            latent_dim_reduction = select_dim_reduction(
                n_dims=dim_latent, default="TSNE", key=1
            )
    input_dim_reducer = fit_dim_reducer(input_dim_reduction, corpus_examples)
    latent_dim_reducer = fit_dim_reducer(latent_dim_reduction, corpus_latents)

    corpus_examples = input_dim_reducer(corpus_examples)
    corpus_latents = latent_dim_reducer(corpus_latents)
    paths = list(map(input_dim_reducer, paths))
    path_latents = list(map(latent_dim_reducer, path_latents))

    st.write("Click on a point to select it for explanation.")
    selection = plot_charts(
        corpus_examples=corpus_examples,
        corpus_latents=corpus_latents,
        n_keep=n_keep,
        selected_index=i,
        explanation_indices=explanation_indices,
        corpus_labels=corpus_labels,
        paths=paths,
        path_latents=path_latents,
    )

    if selection is not None:

        def callback():
            st.session_state.i = selection

        update_streamlit_state(callback=callback)

    visualize_decomposition(
        result=result,
        explanation_indices=explanation_indices,
        n_keep=n_keep,
        selected_index=i,
        feature_names=feature_names,
    )
