#!/bin/bash

python3 plotCumulativeRegret.py --model_id 2 --n_branches 1 --c 1
python3 plotCumulativeRegret.py --model_id 2 --n_branches 1 --c 1 --filenameSufix _more_arms
python3 plotCumulativeRegret.py --model_id 2 --n_branches 1 --c 2
python3 plotCumulativeRegret.py --model_id 2 --n_branches 1 --c 2 --filenameSufix _more_arms