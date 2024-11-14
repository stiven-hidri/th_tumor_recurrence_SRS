from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from utils import Parser
from datasets import ClassifierDataset
from modules import ClassificationModule
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import yaml
import os
import torch

if __name__ == '__main__':
    parser = Parser()

    config, device = parser.parse_args()
    
    # Load sources and create datasets
    classifier_dataset = ClassifierDataset(model_name=config.model.name)
    train_set, val_set, test_set = classifier_dataset.create_split_static()

    # Instantiate Dataloaders for each split
    train_set.augmentation_techniques = []
    train_set.p_augmentation = config.model.p_augmentation
    
    torch.manual_seed(42)
    
    train_dataloader = DataLoader(train_set, batch_size=config.model.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_dataloader = DataLoader(val_set, batch_size=config.model.batch_size, num_workers=4, persistent_workers=True)
    test_dataloader = DataLoader(test_set, batch_size=config.model.batch_size, num_workers=4, persistent_workers=True)

    # Instantiate logger, logs goes into {config.logger.log_dir}/{config.logger.experiment_name}/version_{config.logger.version}
    logger = TensorBoardLogger(save_dir=config.logger.log_dir, version=config.logger.version, name=config.logger.experiment_name )

    # Load pretrained model or else start from scratch
    if config.model.pretrained is None:
        module = ClassificationModule(
            name=config.model.name,
            epochs=config.model.epochs,
            lr=config.model.lr, 
            weight_decay = config.model.weight_decay,
            rnn_type=config.model.rnn_type,
            hidden_size =  config.model.hidden_size,
            num_layers =  config.model.num_layers,
            use_clinical_data=config.model.use_clinical_data,
            alpha_fl = config.model.alpha_fl,
            gamma_fl = config.model.gamma_fl,
            lf = config.model.lf, 
            dropout = config.model.dropout,
            pos_weight = config.model.pos_weight,
            optimizer=config.model.optimizer, 
            scheduler=config.model.scheduler,
            experiment_name=config.logger.experiment_name,
            version=config.logger.version,
            augmentation_techniques = config.model.augmentation_techniques,
            p_augmentation = config.model.p_augmentation,
            depth_attention = config.model.depth_attention
        )
    else:
        module = ClassificationModule.load_from_checkpoint(
            map_location='cpu',
            checkpoint_path=config.model.pretrained,
            name=config.model.name,
            epochs=config.model.epochs,
            lr=config.model.lr,
            rnn_type=config.model.rnn_type,
            hidden_size=config.model.hidden_size,
            num_layers =  config.model.num_layers,
            weight_decay = config.model.weight_decay,
            use_clinical_data=config.model.use_clinical_data,
            alpha_fl = config.model.alpha_fl,
            gamma_fl = config.model.gamma_fl,
            lf = config.model.lf,
            dropout = config.model.dropout,
            pos_weight = config.model.pos_weight,
            optimizer=config.model.optimizer,
            scheduler=config.model.scheduler,
            experiment_name=config.logger.experiment_name,
            version=config.logger.version,
            augmentation_techniques = config.model.augmentation_techniques,
            p_augmentation = config.model.p_augmentation,
            depth_attention = config.model.depth_attention
        )

    # Set callback function to save checkpoint of the model
    checkpoint_cb = ModelCheckpoint(
        monitor=config.checkpoint.monitor,
        dirpath=os.path.join(config.logger.log_dir, config.logger.experiment_name, f'version_{config.logger.version}'),
        filename='{epoch:03d}_{' + config.checkpoint.monitor + ':.6f}',
        save_weights_only=True,
        save_top_k=config.checkpoint.save_top_k,
        mode=config.checkpoint.mode,
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=True, mode="min")

    # Instantiate a trainer
    trainer = Trainer(
        logger=logger,
        accelerator=device,
        devices=[0] if device == "gpu" else "auto",
        default_root_dir=config.logger.log_dir,
        max_epochs=config.model.epochs,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_cb, early_stop_callback],
        log_every_n_steps=1,
        num_sanity_val_steps=0, # Validation steps at the very beginning to check bugs without waiting for training
        reload_dataloaders_every_n_epochs=1,  # Reload the dataset to shuffle the order
    )
    
    # Train
    if not config.model.only_test:
        trainer.fit(model=module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        val_threshold = module.validation_best_threshold
        train_threshold = module.training_best_threshold

    # Get the best checkpoint path and load the best checkpoint for testing
    best_checkpoint_path = checkpoint_cb.best_model_path
    if best_checkpoint_path:
        module = ClassificationModule.load_from_checkpoint(name=config.model.name, checkpoint_path=best_checkpoint_path)

    # Test
    module.validation_best_threshold = val_threshold
    module.training_best_threshold = train_threshold
    trainer.test(model=module, dataloaders=test_dataloader)
