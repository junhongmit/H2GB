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
import datetime
import os
import sys

sys.path.insert(0, os.path.abspath('./'))
import h2gb_sphinx_theme
sys.path.append(os.path.join(os.path.dirname(h2gb_sphinx_theme.__file__), 'extension'))

sys.path.insert(0, os.path.abspath('../../'))
import H2GB

# -- Project information -----------------------------------------------------

author = 'H2GB Team'
project = 'H2GB'
copyright = f'{datetime.datetime.now().year}, {author}'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',
    'h2gb',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme_path = ['.']
html_theme = 'h2gb_sphinx_theme'
html_logo = '_static/logo.png'
html_favicon = '_static/logo.png'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

add_module_names = False
autodoc_member_order = 'bysource'

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    # 'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/dev', None),
    'torch': ('https://pytorch.org/docs/master', None),
}


def rst_jinja_render(app, _, source):
    if hasattr(app.builder, 'templates'):
        rst_context = {'H2GB': H2GB}
        source[0] = app.builder.templates.render_string(source[0], rst_context)


def setup(app):
    app.connect('source-read', rst_jinja_render)
    app.add_js_file('js/version_alert.js')