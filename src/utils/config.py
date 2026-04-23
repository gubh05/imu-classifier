from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    data_dir: Path = Path("data/")
    output_dir: Path = Path("outputs/")
    random_seed: int = 42
    sample_rate_hz: int = 50
    window_size: int = 128
    step_size: int = 32
    cnn_epochs: int = 30
    lstm_epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    activity_labels: dict = field(default_factory=lambda: {
        0: "WALKING",
        1: "WALKING_UPSTAIRS",
        2: "WALKING_DOWNSTAIRS",
        3: "SITTING",
        4: "STANDING",
        5: "LAYING",
    })
