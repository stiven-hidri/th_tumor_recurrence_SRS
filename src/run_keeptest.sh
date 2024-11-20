#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python3 grid_search_whole_dataset.py --config configs/mlp_cd.yaml --experiment_name final_keeptest_mlpcd_minmax --k 10
python3 grid_search_keep_test.py --config configs/base_model.yaml --experiment_name final_keeptest_basemodel_minmax --k 10
python3 grid_search_keep_test.py --config configs/base_model_enhancedV2.yaml --experiment_name final_keeptest_basemodelEV2_minmax --k 10
python3 grid_search_keep_test.py --config configs/wdt_conv.yaml --experiment_name final_keeptest_wdtconv_minmax --k 10
python3 grid_search_keep_test.py --config configs/conv_lstm.yaml --experiment_name final_keeptest_convlstm_minmax --k 10
python3 grid_search_keep_test.py --config configs/trans_med.yaml --experiment_name final_keeptest_transmed_minmax --k 10