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
    
    'dropout':              [.2],
    'weight_decay':         [ 1e-4, 1e-3 ],
    'p_augmentation':       [ .9 ],
    
    'use_clinical_data':    [ False, True ],
    
    'alpha_fl':             [ .2 , .8],
    'gamma_fl':             [2, 3],
}

param_grid_convlstm = {
    'lr':                   [ 1e-4 ],
    'batch_size':           [ 2 ],
    
    'dropout':              [ .1 ],
    'weight_decay':         [ 5e-4 ],
    'p_augmentation':       [ 1. ],
    
    'num_layers':           [ 1, 2 ],
    'hidden_size':          [ 32, 64, 128 ],
    'rnn_type':             [ 'gru', 'lstm' ],
    
    'use_clinical_data':    [ False, True],
    
    'alpha_fl':             [ .7 ],
    'gamma_fl':             [ 2, 3 ],
}

param_grid_wdt = {
    'lr':                   [ 1e-4 ],
    'batch_size':           [ 32 ],
    
    'dropout':              [ .1, .3 ],
    'weight_decay':         [ 1e-4, 1e-3 ],
    'p_augmentation':       [ .9 ] ,
    
    'use_clinical_data':    [ True, False ],
    
    'alpha_fl':             [ .7 ],
    'gamma_fl':             [ 2, 3 ],
}

# todo: 1 patch (eventually try no positional)
param_grid_transmed = {
    'lr':                   [ 1e-4 ],
    'batch_size':           [ 16, 32 ],
    
    'dropout':              [ .0, .1 ],
    'weight_decay':         [ 5e-4 ],
    'p_augmentation':       [ 1. ],
    
    'use_clinical_data':    [ True, False ],
    
    'depth_attention':      [ 6, 8, 10 ],
    
    'alpha_fl':             [ .3, .7 ],
    'gamma_fl':             [ 2, 3 ],   
}