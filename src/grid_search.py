from itertools import product
import re
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils import Parser
from datasets import ClassifierDataset
from modules import ClassificationModule
import os
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import StratifiedGroupKFold
from params_grid import param_grid_basemodel, param_grid_convlstm, param_grid_mlpcd, param_grid_wdt

torch.set_num_threads(8)
torch.cuda.set_per_process_memory_fraction(fraction=.33, device=None)

if __name__ == '__main__':
    # Get configuration arguments
    results_list = []
    parser = Parser()
    config, keep_test, device = parser.parse_args()
    cnt = 0
    version = int(config.logger.version)
    
    if config.model.name == "basemodel":
        param_grid = param_grid_basemodel
    elif config.model.name == "convlstm":
        param_grid = param_grid_convlstm
    elif config.model.name == "mlpcd":
        param_grid = param_grid_mlpcd
    elif config.model.name == "convwdt":
        param_grid = param_grid_wdt
    else:
        raise NotImplementedError("Model not implemented")
    
    # Generate all parameter combinations
    keys, values = zip(*param_grid.items())
    all_param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    max_cnt = len(all_param_combinations)
    
    for i, param_set in enumerate(all_param_combinations):
        print(f"**********\n{i+1}\{max_cnt}:\n**********")
        
        # Load sources and create datasets
        classifier_dataset = ClassifierDataset()
        train_split, val_split, test_split = classifier_dataset.create_splits()
            
        train_dataloader = DataLoader(train_split, batch_size=param_grid['batch_size'], shuffle=True, num_workers=4, persistent_workers=True)
        val_dataloader = DataLoader(val_split, batch_size=param_grid['batch_size'], num_workers=4, persistent_workers=True)
        test_dataloader = DataLoader(test_split, batch_size=param_grid['batch_size'], num_workers=4, persistent_workers=True)

        # Logger for each experiment
        logger = TensorBoardLogger(save_dir=config.logger.log_dir, version=version, name=config.logger.experiment_name)

        del param_grid['batch_size']

        # Create the model with the current hyperparameters
        module = ClassificationModule(
            name =                          config.model.name,
            epochs =                        config.model.epochs,
            lr =                            config.model.lr,
            weight_decay =                  config.model.weight_decay,
            rnn_type =                      config.model.rnn_type,
            hidden_size =                   config.model.hidden_size,
            num_layers =                    config.model.num_layers,
            use_clinical_data =             config.model.use_clinical_data,
            alpha_fl =                      config.model.alpha_fl,
            gamma_fl =                      config.model.gamma_fl,
            lf =                            config.model.lf,
            dropout =                       config.model.dropout,
            pos_weight =                    config.model.pos_weight,
            optimizer =                     config.model.optimizer,
            scheduler =                     config.model.scheduler,
            experiment_name =               config.logger.experiment_name,
            version =                       str(version),
            augmentation_techniques =       config.model.augmentation_techniques,
            p_augmentation =                config.model.p_augmentation,
            p_augmentation_per_technique =  config.model.p_augmentation_per_technique,
            **param_set
        )

        # Checkpoint callback
        checkpoint_cb = ModelCheckpoint(
            monitor=config.checkpoint.monitor,
            dirpath=os.path.join(config.logger.log_dir, config.logger.experiment_name, f'version_{version}'),
            filename='{epoch:03d}_{' + config.checkpoint.monitor + ':.6f}',
            save_weights_only=True,
            save_top_k=config.checkpoint.save_top_k,
            mode=config.checkpoint.mode
        )
        
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min")

        #Trainer
        trainer = Trainer(
            logger=logger,
            accelerator=device,
            devices=[0] if device == "gpu" else "auto",
            default_root_dir=config.logger.log_dir,
            max_epochs=config.model.epochs,
            check_val_every_n_epoch=1,
            callbacks=[checkpoint_cb, early_stop_callback],
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            reload_dataloaders_every_n_epochs=1,
        )

        # Train
        if not config.model.only_test:
            trainer.fit(model=module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # Test

        if checkpoint_cb.best_model_path:
            module = ClassificationModule.load_from_checkpoint(name=config.model.name, checkpoint_path=checkpoint_cb.best_model_path)
        else:
            folder_path = checkpoint_cb.dirpath
            
            pattern = re.compile(r"epoch=(\d+)_val_loss=([\d\.]+)\.ckpt")

            # Variables to keep track of the best checkpoint
            best_val_loss = float('inf')
            best_checkpoint = None

            # Iterate over the files in the folder
            for filename in os.listdir(folder_path):
                match = pattern.match(filename)
                if match:
                    epoch = int(match.group(1))
                    val_loss = float(match.group(2))
                    
                    # Check if current file has a lower validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_checkpoint = filename
            
            path_to_best_checkpoint = os.path.join('log', config.logger.experiment_name, f'version_{version}', best_checkpoint)
            
            module = ClassificationModule.load_from_checkpoint(name=config.model.name, checkpoint_path=path_to_best_checkpoint)
        
        results = trainer.test(model=module, dataloaders=test_dataloader, verbose=False)
        
        result_dict = {
            'version': version,
            **param_set,
            **results[0]  # results[0] is the dictionary returned by the test method
        }
        
        results_list.append(result_dict)
        version += 1

    # Convert the results list to a pandas DataFrame
    df_results = pd.DataFrame(results_list)
    
    df_results.to_excel(os.path.join(os.path.dirname(__file__), 'results_csv', f"{config.logger.experiment_name}.xlsx"), index=False)
