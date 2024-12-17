#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# python3 grid_search_keep_test.py --config configs/mlp_cd.yaml --experiment_name KP_mlpcd_zoom2 --k 5
# python3 grid_search_keep_test.py --config configs/base_model.yaml --experiment_name KP_basemodel_zoom2 --k 5
python3 grid_search_keep_test.py --config configs/wdt_conv.yaml --experiment_name KP_wdtconv_zoom2 --k 5
python3 grid_search_keep_test.py --config configs/conv_lstm.yaml --experiment_name KP_convlstm_zoom2 --k 5
python3 grid_search_keep_test.py --config configs/trans_med.yaml --experiment_name KP_transmed_zoom2 --k 5