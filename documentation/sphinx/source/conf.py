# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.join("..", "..", ".."))

# -- Project information -----------------------------------------------------

project = "AutoKoopman"
copyright = "2022, Ethan Lew"
author = "Ethan Lew"

# The full version, including alpha/beta/rc tags
import autokoopman
release = autokoopman.__version__
version = autokoopman.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc", "sphinx_mdinclude", "sphinx.ext.mathjax"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []


# add this because of https://github.com/pydata/pydata-sphinx-theme/issues/1094
html_theme_options = {
   "github_url": "https://github.com/EthanJamesLew/AutoKoopman",
   "logo": {
      "image_light": "https://raw.githubusercontent.com/EthanJamesLew/AutoKoopman/enhancement/v-0.30-tweaks/documentation/img/brand/logo-small.svg",
      "image_dark": "https://raw.githubusercontent.com/EthanJamesLew/AutoKoopman/enhancement/v-0.30-tweaks/documentation/img/brand/logo-small.svg",
   }
}
