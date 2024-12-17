#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# python3 grid_search_whole_dataset.py --config configs/conv_lstm.yaml --experiment_name WD_convlstm_zoom3 --k 10
python3 grid_search_whole_dataset.py --config configs/trans_med.yaml --experiment_name WD_transmed_zoom3 --k 10
# python3 grid_search_whole_dataset.py --config configs/mlp_cd.yaml --experiment_name WD_mlpcd_minmax_zoom2 --k 10
# python3 grid_search_whole_dataset.py --config configs/wdt_conv.yaml --experiment_name WD_wdtconv_zoom2 --k 10
# python3 grid_search_whole_dataset.py --config configs/base_model.yaml --experiment_name WD_basemodel_zoom2 --k 10