<div align="center">

# 🚆 Station-Wise Railway Delay Prediction using STGCN
### Predicting “snowballing” delays on UK Great Western Railway using Spatio-Temporal Graph Neural Networks

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![GNN](https://img.shields.io/badge/GNN-STGCN-green.svg)]()
[![Graph](https://img.shields.io/badge/Graph-NetworkX-orange.svg)](https://networkx.org/)

🔗 **GitHub Repo:** https://github.com/sumits234/railway_delay_prediction_STGCN

</div>

---

## 📌 Project Overview
Railway networks are complex systems where a small delay at one station can propagate and magnify across the entire network (**snowball effect**). Traditional models often fail to capture **non-linear spatio-temporal dependencies** across the rail network.

This project implements a **Spatio-Temporal Graph Convolutional Network (STGCN)** to predict **link-level delays** (delays on specific track segments) for the **next 10-minute interval**.

✅ Key idea: model railway topology as a **Line Graph**, treating **track links as graph nodes** to capture physical traffic flow.

---

## 🚀 Key Features
- **Graph Formulation:** Converts station network into a **Line Graph (LG)** where nodes represent **track links**
- **Deep Learning Model:** STGCN combining:
  - **Chebyshev Spectral Graph Convolution** for spatial learning
  - **Gated Temporal Convolution** for time-series trends
- **Real-World Dataset:** Trained on ~**104K train service records** from Darwin HSP (2016–2017)
- **Granular Prediction:** **10-minute resolution** prediction on corridor *(Didcot → Paddington)*

---

## 🧠 STGCN Model Summary
- Learns spatial dependencies between adjacent links using graph convolution
- Learns temporal patterns in delays using gated temporal convolution
- Forecasts future delays as a supervised regression task

---

## ⚙️ Workflow / Architecture
```txt
Darwin HSP Logs
     ↓
Data Cleaning + Feature Engineering
     ↓
Line Graph Construction (Links as Nodes)
     ↓
Adjacency Matrix + Time Snapshots
     ↓
STGCN Training (Spatial + Temporal blocks)
     ↓
10-min Ahead Delay Prediction (Link-level)


## 🛠️ Tech Stack
* **Language:** Python 3.8+
* **Deep Learning:** PyTorch
* **Graph Processing:** NetworkX
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
## ⚙️ Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/sumits234/railway_delay_prediction_STGCN.git]
    cd railway_delay_prediction_STGCN
    ```

2.  **Install Dependencies**
    ```bash
    pip install numpy pandas torch networkx matplotlib seaborn scipy
    ```

3.  **Data Preparation**
    Run the notebook to clean raw logs and build the adjacency matrix.
    ```bash
    # Open and run all cells in:
    notebooks/create_data.ipynb
    ```

4.  **Train the Model**
    Execute the training script.
    ```bash
    python main.py --n_epochs 30 --batch_size 100
    ```

## 📊 Results
The model was evaluated on a held-out Test Set (10% of data).

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **MAE** | **0.18 min** | Average prediction error is ~11 seconds. |
| **RMSE** | **0.48 min** | Robust against large outliers. |

**Key Findings:**
* The model successfully predicts the non-linear "snowball" accumulation of delay from Swindon to Reading.
* Rush Hour (Peak) delays were found to be statistically significantly higher than Off-Peak delays.

## 👥 Contributors
* **Sumit Adikari** (22B0615) - *IIT Bombay*

