# Setup and Usage Guide

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Step 1: Clone Repository
```bash
git clone https://github.com/learningnoobi/anamoly-detection.git
cd anamoly-detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Run Complete Demo
```bash
# Run with default settings (15k samples, 20 epochs)
python demo.py

# Custom settings
python demo.py --samples 20000 --epochs 30

# Quick test (no saving results)
python demo.py --samples 5000 --epochs 5 --no-save
```

This will:
1. Generate synthetic network traffic data
2. Preprocess into sequences
3. Train the Transformer-CVAE model
4. Evaluate performance
5. Save results to `./results/` and model to `./checkpoints/`

### Option 2: Step-by-Step Execution

#### 1. Test Data Loading
```bash
python src/load_data.py
```

#### 2. Test Preprocessing
```bash
python src/preprocess.py
```

#### 3. Test Model Architecture
```bash
python src/model.py
```

#### 4. Train Model
```bash
python src/train.py
```

#### 5. Evaluate Model
```bash
python src/evaluate.py
```



## Using Real ToN-IoT Data (work in prod)
