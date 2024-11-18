#Calcola parametri focal loss

param_grid_mlpcd = {
    'batch_size':           [ 32 ],
    
    'lr':                   [ 1e-3, 1e-4 ],
    'dropout':              [ .1, .3, .5 ],
    'weight_decay':         [ 1e-4, 1e-3 ],
    'p_augmentation':       [ .9 ],
    
    'alpha_fl':             [ .2, .8 ],
    'gamma_fl':             [ 2, 3 ]
}

param_grid_basemodel = {
    'lr':                   [5e-4, 1e-4],
    'batch_size':           [32],
    
    'dropout':              [.1, .3],
    'weight_decay':         [ 1e-4, 1e-3 ],
    'p_augmentation':       [ .9 ],
    
    'use_clinical_data':    [ False, True ],
    
    'alpha_fl':             [ .2, .8 ],
    'gamma_fl':             [2, 3],
}


param_grid_wdt = {
    'lr':                   [ 1e-4 ],
    'batch_size':           [ 32 ],
    
    'dropout':              [ .1, .3 ],
    'weight_decay':         [ 1e-4, 1e-3 ],
    'p_augmentation':       [ .9 ] ,
    
    'use_clinical_data':    [ True, False ],
    
    'alpha_fl':             [ .2, .8 ],
    'gamma_fl':             [ 2, 3 ],
}

param_grid_convlstm = {
    'lr':                   [ 1e-5 ],
    'batch_size':           [ 1 ],
    
    'dropout':              [ .1 ],
    'weight_decay':         [ 1e-4 ],
    'p_augmentation':       [ .9 ],
    
    'num_layers':           [ 1, 2 ],
    'hidden_size':          [ 8, 16, 32, 48 ],
    'rnn_type':             [ 'lstm', 'gru' ],
    
    'use_clinical_data':    [ False, True],
    
    'alpha_fl':             [ .2 ],
    'gamma_fl':             [ 2, 3 ],
}

param_grid_transmed = {
    'lr':                   [ 1e-4, 1e-5 ],
    'batch_size':           [ 32 ],
    
    'dropout':              [ .1, .3 ],
    'weight_decay':         [ 1e-4 ],
    'p_augmentation':       [ .9 ],
    
    'use_clinical_data':    [ True, False ],
    
    'depth_attention':      [ 6,8, 10 ],
    
    'alpha_fl':             [ .2 ],
    'gamma_fl':             [ 2, 3 ],   
}