# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

# goes to the root of the project
sys.path.insert(0, os.path.abspath("../../"))

project = "DDQL Optimal Execution"
copyright = "2023, g0bel1n"
author = "g0bel1n"
toc_object_entries_show_parents = "hide"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinxawesome_theme",
]

autodoc_mock_imports = ["torch", "numpy", "pandas", "tqdm"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
}


templates_path = ["_templates"]
exclude_patterns = []

autodoc_default_flags = ["members"]
autosummary_generate = True

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]
