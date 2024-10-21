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

if __name__ == '__main__':
    # Get configuration arguments
    parser = Parser()
    config, device = parser.parse_args()

    # Load sources
    classifier_dataset = ClassifierDataset(p_augmentation=config.model.p_augmentation, augmentation_techniques=config.model.augmentation_techniques)

    # Create train-val-test splits
    train_split, val_split, test_split = classifier_dataset.create_splits()

    # Instantiate Dataloaders for each split
    train_dataloader = DataLoader(train_split, batch_size=config.model.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_dataloader = DataLoader(val_split, batch_size=config.model.batch_size, num_workers=4, persistent_workers=True)
    test_dataloader = DataLoader(test_split, batch_size=config.model.batch_size, num_workers=4, persistent_workers=True)

    # Instantiate logger, logs goes into {config.logger.log_dir}/{config.logger.experiment_name}/version_{config.logger.version}
    logger = TensorBoardLogger(save_dir=config.logger.log_dir, version=config.logger.version, name=config.logger.experiment_name )


    # Load pretrained model or else start from scratch
    if config.model.pretrained is None:
        module = ClassificationModule(
            name=config.model.name,
            epochs=config.model.epochs,
            lr=config.model.learning_rate, 
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
            p_augmentation_per_technique = config.model.p_augmentation_per_technique
        )
    else:
        module = ClassificationModule.load_from_checkpoint(
            map_location='cpu',
            checkpoint_path=config.model.pretrained,
            name=config.model.name,
            epochs=config.model.epochs,
            lr=config.model.learning_rate,
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
            p_augmentation_per_technique = config.model.p_augmentation_per_technique
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

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="min")

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

    # Get the best checkpoint path and load the best checkpoint for testing
    best_checkpoint_path = checkpoint_cb.best_model_path
    if best_checkpoint_path:
        module = ClassificationModule.load_from_checkpoint(name=config.model.name, checkpoint_path=best_checkpoint_path)

    # Test
    results = trainer.test(model=module, dataloaders=test_dataloader)
