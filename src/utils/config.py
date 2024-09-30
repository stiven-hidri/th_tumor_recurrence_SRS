from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional, Union

    
class DatasetConfig(BaseModel):
    name: Optional[str]

class ModelConfig(BaseModel):
    name: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    rnn_type: str
    alpha_fl:float
    gamma_fl: float
    lf:str
    pos_weight:float
    dropout: float
    optimizer: str
    scheduler: str
    only_test: bool
    save_images: Optional[Path] = None
    pretrained: Optional[Path] = None
    annotations: Optional[Path] = None

class LoggerConfig(BaseModel):
    log_dir: Path
    experiment_name: str
    version: int

class CheckpointConfig(BaseModel):
    monitor: str
    save_top_k: int
    mode: str

class Config(BaseModel):
    dataset: DatasetConfig
    model: ModelConfig
    logger: LoggerConfig
    checkpoint: CheckpointConfig
