import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "exp_mpc"
copyright = "2026, Brent Koogler"
author = "Brent Koogler"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_remove_toctrees",
    "myst_parser",
    "sphinx_design",
    # "sphinxext.rediraffe",
    # "source_include",
    # "sphinxcontrib.mermaid",
]

myst_enable_extensions = ["dollarmath", "amsmath"]

autosummary_generate = True
napolean_use_rtype = False

autosummary_imported_members = False
autodoc_default_options = {
    "undoc-members": False,
    "show-inheritance": False,
}

# Type hints in parameter descriptions (not in signatures)
autodoc_typehints = "description"
autodoc_typehints_description_target = "all"

remove_from_toctrees = ["_autosummary/*"]

# NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False
napoleon_use_param = True
napoleon_preprocess_types = True

# Copy button: strip prompts from code blocks
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

html_theme = "sphinx_book_theme"
# html_title = "exp_mpc"
html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/jozbee/mpc_comparison",
    "use_repository_button": True,
    "navigation_with_keys": False,
    "article_header_start": ["toggle-primary-sidebar.html", "breadcrumbs"],
}

html_static_path = ["_static"]
html_css_files = ["style.css"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "flax": ("https://flax.readthedocs.io/en/stable/", None),
    "control": ("https://python-control.readthedocs.io/en/latest/", None),
}

suppress_warnings = [
    "autosummary.import_cycle",
    "app.add_node",
    "myst.xref_missing",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
