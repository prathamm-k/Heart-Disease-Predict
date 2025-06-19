# Heart Disease Risk Analyser

## Project Motivation
I wanted to create a project that helps me in understanding data handling and visulaization. The motivation behind this Heart Disease Risk Analyser was to make myself understand how AI can be used in medical field and how data can bring about a major change in medical industry, and to build a tool that empowers individuals to quickly assess their risk of heart disease using just a few key health indicators. Through this project, I aimed to deepen my understanding of machine learning, data preprocessing, and model evaluation. I learned how to select the most important features, tune and compare models, and create an interactive interface that makes advanced analytics accessible to everyone.

## Project Overview
This project is a machine learning-powered web application that predicts the risk of heart disease based on five essential health features. It leverages a streamlined data science pipeline and a modern, interactive user interface.

**Technologies Used:**
- **Python**: Core language for all data processing, modeling, and app logic.
- **Pandas & NumPy**: For data manipulation and analysis.
- **scikit-learn**: For preprocessing, model building, and evaluation.
- **XGBoost**: For advanced, high-performance tree-based modeling.
- **Streamlit**: To build a fast, interactive, and user-friendly web app.
- **Matplotlib & Plotly**: For visualizing user input distributions and risk.
- **Joblib**: For model serialization and loading.
- **CSV**: For storing the dataset and feature importances.

**How it works:**
- The training pipeline (`train_model/train_model.py`) selects the five most predictive features, preprocesses the data, tunes and compares models, and saves the best one.
- The web app (`app.py`) loads the trained model and lets users enter their health data, instantly showing their risk of any heart disease.

## Project Structure

```
Heart_Disease/
│
├── app.py                      # Streamlit web app for user interaction and prediction
├── requirements.txt            # Python dependencies
│
├── train_model/                # All training and model files
│   ├── train_model.py          # Model training, tuning, and export script
│   ├── heart.csv               # Heart disease dataset (input for training)
│   ├── heart_model_pipeline.joblib   # Trained model pipeline (output)
│   ├── model_features.joblib        # List of features used by the model (output)
│   └── feature_importances.csv      # Feature importances (output)
│
└── heart-venv/                 # (Optional but recommended) Python virtual environment
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/prathamm-k/Heart-Disease-Risk-Analyser.git
cd Heart-Disease-Risk-Analyser
```

### 2. Create and Activate a Virtual Environment

**On Linux/macOS:**
```bash
python3 -m venv heart-venv
source heart-venv/bin/activate
```

**On Windows:**
```bash
python -m venv heart-venv
heart-venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Retrain the Model

If you want to retrain the model (for example, after updating `heart.csv`):

```bash
python train_model/train_model.py
```

This will generate the model and feature files inside the `train_model/` directory.

### 5. Run the Web App

```bash
streamlit run app.py
```

Open the provided local URL in your browser to use the Heart Disease Risk Checker.

## Usage

- Enter your health information in the sidebar.
- Click "Check My Heart Disease Risk".
- Instantly see your estimated risk of heart disease.

## Extra Details

- **Data Privacy:** All predictions are made locally; your data never leaves your computer.
- **Customization:** You can easily extend the project to use more features or a different dataset by updating `train_model.py` and retraining.
- **Model Explainability:** The app uses only the most important features, making predictions transparent and easy to interpret.

## License

This project is open-source and available under the MIT License.

---

*Created by [prathamm-k](https://github.com/prathamm-k)* 