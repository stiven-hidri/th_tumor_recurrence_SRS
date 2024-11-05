param_grid_basemodel = {
    'learning_rate': [1e-4],
    'batch_size': [32],
    'dropout': [.1, .3],
    'weight_decay': [1e-4, 1e-3],
    'gamma_fl': [2, 3],
    'p_augmentation': [.3, .5, .7],
    'use_clinical_data': [False, True]
}

param_grid_convlstm = {
    'learning_rate':        [ 1e-4 ],
    'batch_size':           [ 2 ],
    'dropout':              [ .3 ],
    'weight_decay':         [ 1e-4 ],
    'num_layers':           [ 1, 2 ],
    'hidden_size':          [ 32, 64, 128, 256 ],
    'gamma_fl':             [ 2, 3 ],
    'p_augmentation':       [ .3, .5, .7  ],
    'use_clinical_data':    [ False, True ]
}

param_grid_mlpcd = {
    'learning_rate': [1e-2, 1e-3],
    'batch_size': [16, 32, 64],
    'dropout': [.3, .5],
    'weight_decay': [1e-4, 1e-3],
    'gamma_fl': [2, 3]
}

param_grid_wdt = {
    'learning_rate': [ 0.5e-3 ],
    'batch_size': [ 32 ],
    'dropout': [ .1, .3 ],
    'weight_decay': [ 1e-3, 1e-4 ],
    'p_augmentation': [ .3, .5, .7] ,
    'gamma_fl': [ 2, 3 ],
    'use_clinical_data': [False, True]   
}