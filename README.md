# AI Automation System

## Installation

### 1. Prerequisites
- **[Python 3.9.19](https://www.python.org/downloads/release/python-3919/)**
- **[CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)**
- **[cuDNN 8.9](https://developer.nvidia.com/rdp/cudnn-archive)**
- **Conda** environment

### 2. Environment Setup

Create the environment using Conda and install the required dependencies:

```bash
conda create --name ai_automation_env python=3.9
conda activate ai_automation_env
conda install cudatoolkit=11.8 cudnn=8.9 -c conda-forge
pip install -r requirements.txt
```

Verify TensorFlow installation and GPU availability:

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

## Data Collection

Start recording mouse/keyboard events and screenshots from a specified window:

```bash
python DataSynthesizer.py --window-title "Application Name"
```

Data is saved in the `Database/CollectedTrainingData/` directory.

## Training the Model

Train the model using the collected data:

```bash
python SkillForge.py
```

The model and weights will be saved in `Database/TrainedModel/`.

## Using the Model for Automation

Run the AI controller to automate interactions:

```bash
python main_ai_core.py --window_title "Application Name" --model_json Database/TrainedModel/core.json --model_weights Database/TrainedModel/core.weights.h5
```

This will start the AI automation based on the trained model.
