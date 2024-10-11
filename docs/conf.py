import os
import sys


newpath = os.path.abspath(os.path.join(__file__, '../..'))
print(newpath)
sys.path.insert(0, newpath)
import mdv


project = "Matteo's Python Library"
copyright = 'Matteo Dalle Vedove - 2024'
author = 'Matteo Dalle Vedove'
release = 'v0.0'

extensions = [
    "sphinx_copybutton",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

source_suffix = ['.rst', '.md']

pygments_style = 'sphinx'

html_theme = 'furo'
autosummary_generate = True
