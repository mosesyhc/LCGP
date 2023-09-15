# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'lcgp'
copyright = '2023, Moses Chan'
author = 'Moses Chan'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.mathjax',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon',
              'sphinxcontrib.bibtex',
              'sphinx_copybutton',
              'myst_parser']

bibtex_bibfiles = ['lcgp.bib']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'sphinx_book_theme'

# Side bar elements
html_sidebars = {
    "*": [
        "navbar-logo.html",
        "search-field.html",
        "sbt-sidebar-nav.html"
    ]
}

html_theme_options = {
    "path_to_docs": "docs",
    "show_navbar_depth": 2,
    "repository_url": "https://github.com/mosesyhc/lcgp",
    "use_source_button": True,
    "use_repository_button": True,
    "use_issues_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mosesyhc/lcgp",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/lcgp/",
            "icon": "https://img.shields.io/pypi/dw/lcgp",
            "type": "url",
        },
    ],
}

html_title = "Latent component Gaussian process"
