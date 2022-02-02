import streamlit as st
from streamlit_ace import st_ace


def code_input(
    value: str,
    use_ace_editor=True,
):
    if use_ace_editor:
        return st_ace(
            value=value,
            language="python",
            auto_update=True,
            font_size=12,
        )
    else:
        return st.text_area(
            label="",
            value=value,
            height=400,
        )
