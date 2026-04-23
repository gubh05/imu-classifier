# IMU Sports Activity Classifier

A production-grade, end-to-end pipeline that ingests raw IMU (accelerometer + gyroscope) time-series data, engineers physics-aware features, and classifies six sports/physical activities. This mirrors the core sensor-signal processing problem in smart sports hardware — such as smart footballs or wearables with embedded IMU sensors — and demonstrates real-time inference via a sliding-window simulator.

---

## Architecture

```
Raw IMU Signal (50 Hz, 9 channels)
        │
        ▼
┌───────────────────┐
│  UCIHARDataLoader │  Download, cache, parse raw inertial signals
└────────┬──────────┘
         │  (N, 128, 9)
         ▼
┌───────────────────┐
│ SignalPreprocessor│  Butterworth LPF (20 Hz) + z-score normalisation
└────────┬──────────┘
         │  (N, 128, 9)
         ├─────────────────────────────────┐
         ▼                                 ▼
┌─────────────────┐              ┌──────────────────────┐
│FeatureExtractor │              │  CNN / LSTM Classifier│
│ (RF path only)  │              │  (raw windows)        │
│ 117 features    │              └──────────┬───────────┘
└────────┬────────┘                         │
         ▼                                  │
┌─────────────────┐                         │
│  RFClassifier   │                         │
│  (flat features)│                         │
└────────┬────────┘                         │
         └──────────────┬───────────────────┘
                        ▼
              Predicted Activity Label
                        │
                        ▼
            ┌───────────────────────┐
            │  RealTimeSimulator    │  Sliding-window live inference
            └───────────────────────┘
```

---

## Setup

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate IOTIS-P

# Or create from scratch (Python 3.11)
conda create -n IOTIS-P python=3.11 -y
conda activate IOTIS-P
pip install -r requirements.txt
```

---

## How to Run

### Download data + train all models
```bash
python train.py
```
This will:
1. Download the UCI HAR Dataset automatically (cached after first run)
2. Preprocess signals and extract features
3. Train RF, CNN, and LSTM models
4. Save checkpoints to `outputs/` and plots to `outputs/plots/`
5. Print a results table

### Run the real-time simulator
```python
from src.pipeline.sensor_pipeline import SensorPipeline
from src.simulator.realtime_simulator import RealTimeSimulator
import numpy as np

pipe = SensorPipeline.load("outputs/pipeline_rf.pkl")
sim = RealTimeSimulator(pipe, window_size=128, step_size=32)

# Simulate 5 seconds of IMU data
raw_signal = np.random.randn(250, 9)
for step_idx, label, confidence in sim.stream(raw_signal):
    pass  # table is printed live to stdout
```

### Run tests
```bash
pytest tests/ -v
```

---

## Results

All models evaluated on the UCI HAR test set (2,947 samples, 6 activity classes):

| Model | Accuracy | F1 (weighted) | Train Time |
|-------|----------|---------------|------------|
| Random Forest | 90.02% | 89.97% | 4.6s |
| CNN (1D) | **92.74%** | **92.72%** | 76.9s |
| LSTM | 90.91% | 90.88% | 67.7s |

---

## Example Simulator Output

```
  Step    Time (s)  Activity                Confidence
--------------------------------------------------------
     0       0.00s  WALKING                      72.0%
     1       0.64s  WALKING                      68.5%
     2       1.28s  WALKING_UPSTAIRS             61.0%
     3       1.92s  WALKING_UPSTAIRS             65.5%
     4       2.56s  STANDING                     54.0%
     5       3.20s  STANDING                     71.0%
```

---

## Design Decisions

**Why Butterworth low-pass filter?**  
IMU accelerometer signals contain high-frequency noise from vibration and quantisation. A 4th-order Butterworth filter at 20 Hz (Nyquist = 25 Hz at 50 Hz sampling rate) removes this noise with a maximally flat passband, preserving the motion dynamics relevant to activity recognition without introducing phase distortion artefacts common in other filter types.

**Why both CNN and LSTM?**  
These architectures capture temporal structure in fundamentally different ways: the 1D-CNN learns local motif patterns (e.g., a step's acceleration spike) via convolutional kernels, while the LSTM models long-range sequential dependencies across the full 2.56-second window. Comparing both reveals which temporal structure is more discriminative for this task — a key design insight for deploying on embedded hardware where model complexity must be justified.

**Why Random Forest as a baseline?**  
RF operates on hand-engineered features (time + frequency domain per channel), making it fully interpretable via feature importances. It serves as a strong, fast baseline that validates the feature engineering pipeline independently of deep learning, and its `feature_importances_` output guides which signal characteristics are most discriminative — informing both model design and potential sensor-channel pruning for hardware cost reduction.
