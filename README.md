# Experimental MPC

We implement some hacky (python) code to compute and visualize MPC
implementations of Stewart platforms.
The code can be used as a Python library or as a C++ library for real-time
applications.
See the docs for more information.
To build to docs, first install the necessary dependencies, i.e., in the root
git directory, run

```bash
pip install -e ".[docs]"
```

Then build the docs by running

```bash
sphinx-build -j auto -b html docs docs/_build/html
```

To view the docs, open `docs/_build/html/index.html` in a webbrowser.
