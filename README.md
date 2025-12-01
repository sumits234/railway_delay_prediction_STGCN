# Station-Wise Railway Delay Prediction using STGCN

**A Spatio-Temporal Graph Convolutional Network (STGCN) approach to predict "snowballing" delays on the UK Great Western Railway.**

## 📌 Project Overview
Railway networks are complex systems where a small delay at one station can propagate and magnify across the entire network (the "snowball effect"). Traditional statistical models often fail to capture the complex, non-linear spatial dependencies between tracks.

This project implements a **Spatio-Temporal Graph Convolutional Network (STGCN)** to predict **link-level delays** (delays on specific track segments) for the next 10-minute interval. It explicitly models the railway topology as a **Line Graph**, treating track segments as nodes to capture the physical flow of traffic.

## 🚀 Key Features
* **Graph Formulation:** Converts physical railway stations into a **Line Graph (LG)** where nodes represent track links.
* **Deep Learning Model:** Uses **Chebyshev Spectral Graph Convolution** for spatial features and **Gated Temporal Convolution** for time-series trends.
* **Real-World Data:** Trained on ~104,000 service records from the UK's Darwin HSP system (2016-2017).
* **Granular Prediction:** Forecasts delays with a **10-minute resolution** along the high-density Didcot \to Paddington corridor.

## 📂 Dataset
* **Source:** Darwin Historical Service Performance (HSP) API.
* **Route:** Great Western Railway (GWR) Inbound: *Swansea -> Cardiff -> Bristol -> Swindon -> Didcot -> Reading -> London Paddington*.
* **Volume:** 104,012 train stop records processed into **21,336 time snapshots** (T=12 input window).
* **Features:** Link Delay, Link Speed, Time of Day (Cyclical), Day of Week.

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
