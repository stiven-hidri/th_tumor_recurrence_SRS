#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# python3 grid_search_whole_dataset.py --config configs/mlp_cd.yaml --experiment_name final_wholedataset_mlpcd_minmax --k 10
# python3 grid_search_whole_dataset.py --config configs/base_model.yaml --experiment_name final_wholedataset_basemodel_minmax --k 10
python3 grid_search_whole_dataset.py --config configs/base_model_enhancedV2.yaml --experiment_name final_wholedataset_basemodelEV2_equaldeeper --k 10
python3 grid_search_whole_dataset.py --config configs/wdt_conv.yaml --experiment_name final_wholedataset_wdtconv_scaledFC --k 10
python3 grid_search_whole_dataset.py --config configs/conv_lstm.yaml --experiment_name final_wholedataset_convlstm_deeper --k 10
python3 grid_search_whole_dataset.py --config configs/trans_med.yaml --experiment_name final_wholedataset_transmed_PS2 --k 10