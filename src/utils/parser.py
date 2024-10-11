import os
import yaml
import torch
from argparse import ArgumentParser
from pathlib import Path
from utils.config import Config


class Parser:
    def __init__(self):
        self.parser = ArgumentParser()

        # General mode args
        self.parser.add_argument('-c', '--config', type=Path, help='Configuration Path', required=True, dest='CONFIG')
        self.parser.add_argument('--cpu', help='Set CPU as device', action='store_true', dest='CPU')

        # General data args
        self.parser.add_argument('--lr', type=float, help='Learning rate', dest='LR')
        self.parser.add_argument('--weight_decay', type=float, help='Weight decay', dest='WEIGHT_DECAY')
        self.parser.add_argument('--rnn_type', type=float, help='Weight decay', dest='RNN_TYPE')
        self.parser.add_argument('--alpha_fl', type=float, help='Alpha focal loss', dest='ALPHA_FL')
        self.parser.add_argument('--gamma_fl', type=float, help='Gamma focal loss', dest='GAMMA_FL')
        self.parser.add_argument('--lf', type=float, help='Loss function acronym', dest='LF')
        self.parser.add_argument('--dropout', type=float, help='Dropout', dest='DROPOUT')
        self.parser.add_argument('--pos_weight', type=float, help='Positive weight WBCE', dest='POS_WEIGHT')
        self.parser.add_argument('--epochs', type=int, help='Number of epochs', dest='EPOCHS')
        self.parser.add_argument('--pretrained', type=str, help='Path to pretrained model checkpoint', dest='PRETRAINED')
        self.parser.add_argument('--only_test', help='Set True to skip training', action='store_true', dest='ONLY_TEST')
        self.parser.add_argument('--augmentation_techniques', type=list, help='Augmentation techinques to apply', dest='augmentation_techniques')
        self.parser.add_argument('--p_augmentation', type=float, help='Probability to augment', dest='P_AUGMENTATION')
        self.parser.add_argument('--p_augmentation_per_technique', type=float, help='Probability to apply each technique', dest='P_AUGMENTATION_PER_TECHNIQUE')

        # Logger args
        self.parser.add_argument('--experiment_name', type=str, help='Experiment name', dest='EXPERIMENT_NAME')
        self.parser.add_argument('--version', type=int, help='Version number', dest='VERSION_NUMBER')

    def parse_args(self) -> [Config, str]:
        self.args = self.parser.parse_args()

        # Resolve Warning: 
        # oneDNN custom operations are on. 
        # You may see slightly different numerical results due to floating-point round-off errors from different computation orders. 
        # To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

        # Check if CUDA devices are available
        available_devs = torch.cuda.device_count()
        if self.args.CPU:
            device = 'cpu'
        else:
            if available_devs >= 1:
                device = 'gpu'
            else:
                print('Couldn\'t find a GPU device, running on cpu...')
                device = 'cpu'

        with open(self.args.CONFIG) as f:
            d = yaml.safe_load(f)
            config = Config(**d)

        if self.args.LR is not None:
            config.model.learning_rate = self.args.LR

        if self.args.WEIGHT_DECAY is not None:
            config.model.weight_decay = self.args.WEIGHT_DECAY
            
        if self.args.RNN_TYPE is not None:
            config.model.rnn_type = self.args.RNN_TYPE
            
        if self.args.ALPHA_FL is not None:
            config.model.alpha_fl= self.args.ALPHA_FL
            
        if self.args.GAMMA_FL is not None:
            config.model.gamma_fl = self.args.GAMMA_FL
            
        if self.args.LF is not None:
            config.model.lf = self.args.LF
            
        if self.args.DROPOUT is not None:
            config.model.dropout = self.args.DROPOUT
            
        if self.args.POS_WEIGHT is not None:
            config.model.pos_weight = self.args.POS_WEIGHT

        if self.args.EPOCHS is not None:
            config.model.epochs = self.args.EPOCHS

        if self.args.PRETRAINED is not None:
            config.model.pretrained = self.args.PRETRAINED

        if self.args.ONLY_TEST is not None:
            config.model.only_test = self.args.ONLY_TEST
        
        if self.args.EXPERIMENT_NAME is not None:
            config.logger.experiment_name = self.args.EXPERIMENT_NAME

        if self.args.VERSION_NUMBER is not None:
            config.logger.version = self.args.VERSION_NUMBER
            
        if self.args.augmentation_techniques is not None:
            config.logger.augmentation_techniques = self.args.augmentation_techniques
            
        if self.args.P_AUGMENTATION is not None:
            config.logger.p_augmentation = self.args.P_AUGMENTATION
            
        if self.args.P_AUGMENTATION_PER_TECHNIQUE is not None:
            config.logger.p_augmentation_per_technique = self.args.P_AUGMENTATION_PER_TECHNIQUE

        return config, device
