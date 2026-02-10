"""Sphinx configuration for FaultMap documentation."""

project = "FaultMap"
copyright = "2024, Simon Streicher"
author = "Simon Streicher"
release = "0.1.3"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
}

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
