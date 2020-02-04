#!/usr/bin/env bash

rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pre-commit install