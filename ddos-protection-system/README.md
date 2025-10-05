# An Adaptive, Machine Learning-Based DDoS Protection System for Cloud Infrastructure

## Project Objective

The primary objective of this project is to design, implement, and evaluate a robust, multi-stage DDoS mitigation framework specifically for cloud environments. The system leverages a hybrid machine learning model to accurately detect both known and novel (zero-day) DDoS attacks in real-time. The goal is to create a scalable, cloud-native solution that can automatically analyze network traffic, identify threats with high precision, and trigger automated remediation actions to ensure service availability and security.

## Core System Architecture & Modules

The system is composed of three primary stages:

### Module 1: Real-time Traffic Monitoring and Feature Extraction

*   **Functionality:** Ingests network traffic data from various sources (e.g., log files, real-time packet capture).
*   **Processing:** Uses a data streaming pipeline to process high-velocity data.
*   **Feature Engineering:** Extracts a comprehensive set of features, including flow-based, time-based statistical, and behavioral features to create a rich representation of the traffic for the ML models.

### Module 2: Hybrid Detection Engine

*   **Core Logic:** Employs a dual-model approach for defense-in-depth.
*   **Component A (Supervised): XGBoost Classifier:** Trained on a labeled dataset (CICDDoS2019) to accurately identify known DDoS attack patterns with high speed and precision.
*   **Component B (Unsupervised): LSTM Autoencoder:** Trained only on benign traffic to detect novel anomalies and zero-day attacks by identifying high reconstruction errors.
*   **Ensemble Logic:** Combines the outputs of both models to make a final decision (Benign/Malicious) and trigger alerts.

### Module 3: Automated Remediation and Policy Enforcement

*   **Functionality:** Receives alerts from the Detection Engine and executes automated responses.
*   **Actions:** Implements a range of mitigation strategies, including:
    *   **IP Filtering:** Blocking malicious source IPs using firewall rules.
    *   **Rate Limiting:** Throttling traffic from suspicious sources to mitigate application-layer attacks.
    *   **Alerting:** Notifying administrators of the detected threat and the actions taken.

## Technical Stack

*   **Programming Language:** Python 3.8+
*   **Dataset:** CICDDoS2019
*   **Data Processing & Analysis:** Pandas, NumPy, Scikit-learn (for preprocessing)
*   **Machine Learning Models:**
    *   XGBoost Library for the supervised classifier.
    *   TensorFlow/Keras for the LSTM Autoencoder.
*   **API & Backend:** Flask to create an API endpoint for the detection engine.
*   **Deployment (Optional but Recommended):** Docker for containerizing the application components.

## Project Structure

```
ddos-protection-system/
│
├── data/
│   ├── raw/
│   │   └── CICDDoS2019.csv       # The original, unprocessed dataset (or placeholder)
│   └── processed/
│       ├── train_data_features.csv # Preprocessed and balanced training features
│       ├── train_data_labels.csv  # Preprocessed and balanced training labels
│       ├── test_data_features.csv # Preprocessed testing features
│       └── test_data_labels.csv   # Preprocessed testing labels
│
├── notebooks/
│   ├── 1_data_exploration.ipynb   # Jupyter notebook for EDA
│   └── 2_model_prototyping.ipynb  # Notebook for initial model building and testing
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Script with functions for cleaning and preparing data
│   ├── model_training.py          # Script to train and save the XGBoost and LSTM models
│   ├── predict.py                 # Core logic for the hybrid detection engine
│   └── app.py                     # Flask/FastAPI application to serve the model via API
│
├── models/
│   ├── xgboost_classifier.pkl     # Saved trained XGBoost model
│   ├── lstm_autoencoder.h5        # Saved trained LSTM Autoencoder model
│   ├── minmax_scaler.pkl          # Saved MinMaxScaler for preprocessing new data
│   ├── feature_selector.pkl       # Saved RFE selector for feature selection
│   └── anomaly_threshold.pkl      # Saved anomaly threshold for LSTM Autoencoder
│
├── scripts/
│   └── remediation.py             # Contains functions for simulated mitigation actions
│
├── tests/
│   └── test_api.py                # Script to test the prediction API endpoint
│
├── requirements.txt               # List of all Python dependencies for the project
└── README.md                      # Project documentation: setup, usage, and results
```

## Step-by-Step Implementation Plan

### Phase 1: Environment Setup and Data Preparation

1. **Setup Virtual Environment:** Create a Python virtual environment to manage project dependencies.
   ```bash
   python -m venv ddos-protection-system/venv
   ```
2. **Install Libraries:** Install all the libraries listed in the `requirements.txt`.
   ```bash
   ddos-protection-system/venv/Scripts/pip install -r ddos-protection-system/requirements.txt
   ```
3. **Download Dataset:** Obtain the CICDDoS2019 dataset from the Canadian Institute for Cybersecurity. Place the `.csv` file(s) in `data/raw/`.
4. **Data Exploration (EDA):** Load the dataset using Pandas. Analyze the features, understand the distribution of attack types, and identify data quality issues.
5. **Data Preprocessing:**
   * Handle missing or infinite values by imputing them (e.g., with the mean).
   * Normalize numerical features to a range of [0, 1] using Min-Max scaling. This is crucial for the LSTM Autoencoder.
   * Select the most impactful features using a technique like Recursive Feature Elimination (RFE) to reduce dimensionality.
   * Split the data into training (80%) and testing (20%) sets.
   * Apply the SMOTE technique to the *training data only* to handle class imbalance.
   * The `src/data_preprocessing.py` script contains the necessary functions.

### Phase 2: Model Development and Training

1. **Develop XGBoost Model:**
   * Implement an XGBoost classifier.
   * Train it on the preprocessed, balanced training data.
   * Perform hyperparameter tuning to optimize its performance.

2. **Develop LSTM Autoencoder Model:**
   * Design a neural network architecture using LSTM layers.
   * Train this model *only on the benign samples* from the training dataset.
   * Determine an appropriate threshold for the reconstruction error to classify traffic as anomalous.

3. **Model Evaluation:**
   * Evaluate both models on the unseen test set.
   * Calculate and report the key performance metrics: **Accuracy, Precision, Recall, and F1-Score**.
   * Save the trained models (e.g., as `.pkl`, `.h5`, or `.pt` files) for later use.
   * The `src/model_training.py` script handles model training and saving.

### Phase 3: Building the Full Detection Pipeline

1. **Create the Ensemble Logic:** The `src/predict.py` script contains the `HybridDDoSDetector` class which encapsulates the ensemble logic, combining predictions from both XGBoost and LSTM Autoencoder.

2. **Build a Web API:** The `src/app.py` file implements a Flask application with a `/predict` endpoint that accepts network traffic features in JSON format and returns the classification result.
   * To run the API: `python ddos-protection-system/src/app.py`

3. **Implement Remediation Scripts:** The `scripts/remediation.py` file contains functions that simulate remediation actions, such as blocking IPs, rate limiting, and sending alerts.

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ARULAMUTHAN1/DDOS.git
   cd ddos-protection-system
   ```

2. **Setup Virtual Environment and Install Dependencies:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # on Windows
   source venv/bin/activate # on Linux/macOS
   pip install -r requirements.txt
   ```

3. **Download Dataset:**
   Download the CICDDoS2019 dataset and place the `.csv` files into the `data/raw/` directory.

4. **Run Data Preprocessing and Model Training:**
   Execute the `notebooks/2_model_prototyping.ipynb` notebook to preprocess the data and train the models. **This step is crucial as it generates the model files (`.pkl`, `.h5`) and preprocessing artifacts (`.pkl`) in the `models/` directory.**
   
   Alternatively, you can run the scripts directly:
   ```bash
   python src/data_preprocessing.py  # if you want to run preprocessing separately
   python src/model_training.py      # This will call preprocessing if needed
   ```

5. **Start the API Server:**
   ```bash
   python src/app.py
   ```
   The API will be accessible at `http://127.0.0.1:5000`.

6. **Test the API:**
   Run the test script: `python tests/test_api.py`
   This script will send sample requests to the running API and verify responses.

## Evaluation Criteria

The success of the project will be measured by:

* **Model Performance:** Achieving high scores for **Accuracy (target >97%)**, **Precision**, **Recall**, and **F1-Score** on the test dataset, benchmarked against the results in the paper.
* **Zero-Day Detection:** Demonstrating the LSTM Autoencoder's ability to flag anomalous data not seen during training.
* **System Functionality:** A working API that correctly classifies incoming data and triggers the appropriate (simulated) remediation response.
* **Code Quality:** Clean, well-documented, and organized code following the proposed file structure.