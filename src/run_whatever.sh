#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python3 grid_search_keep_test.py --config configs/mlp_cd.yaml --experiment_name KP_mlpcd --k 5
python3 grid_search_keep_test.py --config configs/base_model.yaml --experiment_name KP_basemodel --k 5

python3 grid_search_keep_test.py --config configs/wdt_conv.yaml --experiment_name KP_wdtconv --k 5
python3 grid_search_keep_test.py --config configs/conv_lstm.yaml --experiment_name KP_convlstm --k 5
python3 grid_search_keep_test.py --config configs/trans_med.yaml --experiment_name KP_transmed --k 5

python3 grid_search_whole_dataset.py --config configs/mlp_cd.yaml --experiment_name WD_mlpcd --k 10
python3 grid_search_whole_dataset.py --config configs/base_model.yaml --experiment_name WD_basemodel --k 10

python3 grid_search_whole_dataset.py --config configs/wdt_conv.yaml --experiment_name WD_wdtconv --k 10
python3 grid_search_whole_dataset.py --config configs/conv_lstm.yaml --experiment_name WD_convlstm --k 10
python3 grid_search_whole_dataset.py --config configs/trans_med.yaml --experiment_name WD_transmed --k 10