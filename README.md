# CausalPD-main

> **Joint Causal Discovery and Intervention for Large-Scale Pavement Distress Distribution Data**

---

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Introduction

**CausalPD-main** is a deep learning framework for joint causal discovery and intervention on large-scale pavement distress distribution data. It provides tools for data loading, model training, evaluation, and visualization, aiming to facilitate research in spatio-temporal causal inference and pavement management.


---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/xuesong-wu/CausalPD-main.git
   cd CausalPD-main
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Prepare your data

- Place your data in the `data/` directory. See [Data Preparation](#data-preparation) for details.

### 2. Run experiments

- Example (training & evaluation):
  ```bash
  python run_Exp.py --is_training 1 --model_id test --model CausalPD --data train --root_path ./data/GridSZ/ --data_path pavement_distress.npy
  ```

- For more options, see the arguments in `run_Exp.py`.

### 3. One-click training for each dataset

- You can use the provided shell scripts in the `scripts/` directory to quickly start training for each dataset:

  ```bash
  # Train on GridSZ dataset
  bash scripts/GridSZ.sh

  # Train on SegmentSZ dataset
  bash scripts/SegmentSZ.sh

  # Train on Shanghai dataset
  bash scripts/Shanghai.sh
  ```

- Each script contains recommended parameters for the corresponding dataset. You can modify the scripts to adjust hyperparameters as needed.

---

## Project Structure

```
CausalPD-main/
│
├── data/                # Datasets (not included, see Data Preparation)
├── data_provider/       # Data loading and preprocessing
├── exp/                 # Experiment logic
├── layers/              # Model layers and backbone
├── models/              # Model definitions
├── utils/               # Utility functions
├── results/             # Model outputs (ignored by git)
├── checkpoints/         # Model checkpoints (ignored by git)
├── run_Exp.py           # Main entry point for training/testing
├── requirements.txt     # Python dependencies
└── .gitignore           # Git ignore rules
```

---

## Data Preparation

- **Data is not included in this repository.**
- Please place your dataset files (e.g., `.npy`, `.csv`) in the `data/` directory.
- Example structure:
  ```
  data/
    ├── GridSZ/
    │     ├── pavement_distress.npy
    │     ├── ext.csv
    │     └── ...
    ├── SegmentSZ/
    │     ├── pavement_distress.npy
    │     └── ...
    └── ...
  ```
- Download datasets:
  - [GridSZ (Google Drive)](https://drive.google.com/file/d/1I-TudjLAI0siukg-1UgPC798SkCA2auG/view?usp=sharing)
  - [SegmentSZ (Google Drive)](https://drive.google.com/file/d/1PpAzvXrrxbBK5PyAO033B748CnNa21ae/view?usp=sharing)


---

## Results

- Model predictions and evaluation results will be saved in the `results/` and `test_results/` directories.
- Visualization outputs (e.g., `.pdf` plots) are also stored here.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for suggestions, bug reports, or improvements.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or collaboration, please contact:

- **Author:** Xuesong Wu  
- **GitHub:** [xuesong-wu](https://github.com/xuesong-wu) 