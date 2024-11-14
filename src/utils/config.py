from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional, Union

    
class DatasetConfig(BaseModel):
    name: Optional[str]

class ModelConfig(BaseModel):
    name: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    rnn_type: str
    hidden_size: int
    num_layers: int
    alpha_fl:float
    use_clinical_data:bool
    gamma_fl: float
    lf:str
    pos_weight:float
    dropout: float
    optimizer: str
    scheduler: str
    only_test: bool
    augmentation_techniques: list
    p_augmentation: float
    depth_attention : float
    save_images: Optional[Path] = None
    pretrained: Optional[Path] = None
    annotations: Optional[Path] = None
    

class LoggerConfig(BaseModel):
    log_dir: Path
    experiment_name: str
    version: int
    keep_test: Optional[bool] = False
    k: Optional[int] = 6
    majority_vote: Optional[bool] = False

class CheckpointConfig(BaseModel):
    monitor: str
    save_top_k: int
    mode: str

class Config(BaseModel):
    dataset: DatasetConfig
    model: ModelConfig
    logger: LoggerConfig
    checkpoint: CheckpointConfig
