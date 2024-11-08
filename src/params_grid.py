param_grid_mlpcd = {
    'lr':                   [ 1e-2, 1e-3 ],
    'batch_size':           [ 16, 32 ],
    'dropout':              [ .3, .5 ],
    'weight_decay':         [ 1e-4, 1e-3 ],
    'gamma_fl':             [ 2, 3 ]
}

param_grid_basemodel = {
    'lr':                   [1e-4],
    'batch_size':           [32],
    'dropout':              [.1, .3],
    'weight_decay':         [1e-4, 1e-3],
    'gamma_fl':             [2, 3],
    'p_augmentation':       [.3, .5, .7],
    'use_clinical_data':    [True, False]
}

param_grid_convlstm = {
    'lr':                   [ 1e-4 ],
    'batch_size':           [ 2 ],
    'dropout':              [ .3 ],
    'weight_decay':         [ 1e-4 ],
    'num_layers':           [ 2 ],
    'hidden_size':          [ 32, 64, 128 ],
    'gamma_fl':             [ 2, 3 ],
    'p_augmentation':       [ .3, .5, .7  ],
    'use_clinical_data':    [ True, False ]
}

param_grid_wdt = {
    'lr':                   [ 0.5e-3 ],
    'batch_size':           [ 32 ],
    'dropout':              [ .1, .3 ],
    'weight_decay':         [ 1e-3, 1e-4 ],
    'p_augmentation':       [ .3, .5, .7 ] ,
    'gamma_fl':             [ 2, 3 ],
    'use_clinical_data':    [ True, False ]   
}

param_grid_transmed = {
    'lr':                   [ 1e-3 ],
    'batch_size':           [ 4 ],
    'dropout':              [ .1 ],
    'weight_decay':         [ 1e-4 ],
    'p_augmentation':       [ 0. ] ,
    'gamma_fl':             [ 2, 3 ],
    'use_clinical_data':    [ False ]   
}