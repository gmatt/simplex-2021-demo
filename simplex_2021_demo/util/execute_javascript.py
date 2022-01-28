import time

import streamlit.components.v1 as components


def execute_javascript(
    code: str,
):
    """Executes JavaScript code inside the Streamlit app."""
    # Set the document to the one outside the iframe, since usually this is the one we use.
    # Add timestamp because otherwise Streamlit would run the script only once.
    code = f"""(() => {{
const document = parent.document;
{code}
}})();
// {time.time()}"""
    components.html(f"<script>\n{code}\n</script>", height=0)
