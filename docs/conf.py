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
#import os
#import sys
import ktorch_sphinx_theme

# -- Project information -----------------------------------------------------
project = 'ktorch'
copyright = '2022'
author = 'ktorch'

# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxcontrib.katex',
    'sphinx.ext.autosectionlabel',
]

autosummary_generate = True
autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

#html_theme = 'alabaster'
html_theme = 'ktorch_sphinx_theme'
html_theme_path = [ktorch_sphinx_theme.get_html_theme_path()]
html_theme_options = {
    'ktorch_project': 'docs',
    'canonical_url': 'https://ktorch.readthedocs.io/en/latest/',
    'collapse_navigation': False,
    'display_version': False,
    'logo_only': True,
}
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]
master_doc = 'index'
