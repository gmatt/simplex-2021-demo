from typing import Callable

import streamlit as st

from simplex_2021_demo.util.execute_javascript import execute_javascript

BUTTON_LABEL = "loading"


def update_streamlit_state(
    callback: Callable,
):
    # Create a button, click it from JavaScript, then immediately hide it.
    # I wish there was a less silly solution, but I couldn't yet find one.

    execute_javascript(
        # language=js
        """
const interval = setInterval(() => {
    const button = [...document.querySelectorAll('.stButton > button')]
        .find(b => b.textContent === '%(BUTTON_LABEL)s');
    if (button != null) {
        button.click();
        clearInterval(interval);
    }
}, 10);
"""
        % {"BUTTON_LABEL": BUTTON_LABEL}
    )

    empty = st.empty()
    if empty.button(
        label=BUTTON_LABEL,
        on_click=callback,
    ):
        empty.write("")
