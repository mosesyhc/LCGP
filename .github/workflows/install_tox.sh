#!/bin/bash

python -m pip install --upgrade pip
python -m pip install --upgrade setuptools
python -m pip install --upgrade wheel
python -m pip install --upgrade tox
python -m pip install --upgrade ruff
echo
which python
which pip
which tox
which ruff
echo
python --version
pip --version
tox --version
ruff --version
echo
pip list
