# AI Network Traffic Analysis Project
(Due to size limit issue)
PROJECT FOLDER DRIVE LINK : https://drive.google.com/drive/folders/1jNjLN_Vsu1YOl-CVvC1Kfkb72o3_yZfS?usp=drive_link

## 🚀 Project Overview

This project implements an **AI-powered Network Traffic Classification System** that can detect and classify different types of network traffic, including normal traffic and various types of attacks. The system uses machine learning algorithms to analyze network traffic patterns and provide real-time classification capabilities.

### 🎯 Key Features

- **Multi-Dataset Support**: Works with CICIDS-2017 and ISCX VPN-NonVPN datasets
- **Advanced ML Models**: Implements Random Forest and XGBoost classifiers
- **Web Interface**: User-friendly Flask web application for real-time predictions
- **Processing**: CSV file predictions
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations
- **Feature Selection**: Automatic feature selection for optimal model performance

### 📊 Supported Traffic Types

**CICIDS-2017 Dataset:**
- Normal Traffic
- DDoS Attacks
- PortScan Attacks

**ISCX VPN-NonVPN Dataset:**
- VPN Traffic
- Non-VPN Traffic
- Various application types (Chat, Email, File Transfer, etc.)

## 📁 Project Structure

```
ai_network_traffic_project/
├── app.py                          # Flask web application
├── train_models.py                 # Model training script
├── predict.py                      # Prediction utilities
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/                           # Dataset links and information
│   ├── CICIDS_Traffic_Dataset/
│   └── Encypted_Traffic_Dataset/
├── models/                         # Trained model files
│   ├── cicids_model.pkl
│   └── encrypted_model.pkl
├── processed/                      # Preprocessed datasets
├── preprocessing/                  # Data preprocessing scripts
├── results/                        # Model evaluation results
└── templates/                      # Web interface templates
```

## 🔗 External Resources

Due to GitHub's file size limitations, the complete project with datasets, reports, and videos is available on Google Drive:

### 📂 Complete Project Folder
**Google Drive Link**: https://drive.google.com/drive/folders/1jNjLN_Vsu1YOl-CVvC1Kfkb72o3_yZfS?usp=drive_link


The drive folder contains:
- 📊 **Original Datasets**: CICIDS-2017 and ISCX VPN-NonVPN (Links of datasets)
- 📈 **Complete Results**: All evaluation metrics, confusion matrices, and performance plots
- 📋 **Project Report**: Detailed analysis and findings (drive link provided)
- 🎥 **Execution Video**: Step-by-step demonstration of the system (drive link provided)
- 📊 **Presentation**: PowerPoint slides explaining the project (drive link provided)

### 📊 Dataset Links

**CICIDS-2017 Dataset**: https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

**ISCX VPN Dataset**: http://cicresearch.ca/CICDataset/ISCX-VPN-NonVPN-2016/
## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd ai_network_traffic_project
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Datasets

1. Download the datasets from the provided links above
2. Place them in the `data/` directory
3. Run the preprocessing script to prepare the data:

```bash
python preprocessing/prepare_datasets.py
```

### Step 4: Train the Models

```bash
python train_models.py
```

This will:
- Load and preprocess the datasets
- Train Random Forest and XGBoost models
- Perform hyperparameter tuning
- Generate evaluation metrics and visualizations
- Save trained models to `models/` directory (XgBoost is the final saved model)

## 🚀 Usage Guide

### Option 1: Web Interface (Recommended)

1. **Start the Flask Application**:
   ```bash
   python app.py
   ```

2. **Open Your Browser**:
   Navigate to `http://localhost:5000`

3. **Use the Web Interface**:
   - Select the dataset type (CICIDS or Encrypted)
   - Choose prediction method:
     - **CSV Upload**: Upload a CSV file with network traffic data
   - View real-time predictions and confidence scores

### Option 2: Command Line Interface


#### Batch Prediction from CSV

```bash
python predict.py --mode batch --dataset cicids --input_file your_data.csv --output_file predictions.csv
```

#### Model Evaluation

```bash
python predict.py --mode evaluate --dataset cicids --input_file test_data.csv
```

## 📊 Model Performance

### CICIDS-2017 Dataset Results
- **Accuracy**: ~99.5%
- **Precision**: ~99.4%
- **Recall**: ~99.5%
- **F1-Score**: ~99.4%

### ISCX VPN-NonVPN Dataset Results
- **Accuracy**: ~98.2%
- **Precision**: ~98.1%
- **Recall**: ~98.2%
- **F1-Score**: ~98.1%

## 🔧 Configuration

### Model Parameters

The models use optimized hyperparameters:

**Random Forest:**
- n_estimators: 100-200
- max_depth: 10-15
- min_samples_split: 2-5
- class_weight: 'balanced'

**XGBoost:**
- n_estimators: 100-150
- max_depth: 3-7
- learning_rate: 0.1-0.2
- subsample: 0.8-0.9

### Feature Selection

- **CICIDS Dataset**: 30 most important features selected
- **Encrypted Dataset**: 30 most important features selected
- Selection method: ANOVA F-test (f_classif)

## 📈 Results and Visualizations

The training process generates comprehensive results in the `results/` directory:

- **Confusion Matrices**: Visual representation of classification performance
- **Feature Importance Plots**: Shows which features contribute most to predictions
- **Performance Metrics**: Detailed accuracy, precision, recall, and F1-scores
- **Class-wise Analysis**: Performance breakdown by traffic type

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Error**:
   ```bash
   # Ensure models are trained first
   python train_models.py
   ```

2. **Missing Dependencies**:
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt
   ```

3. **Dataset Not Found**:
   - Download datasets from provided links
   - Place in correct directories
   - Run preprocessing script

4. **Memory Issues**:
   - Reduce dataset size for testing
   - Use smaller feature selection (modify `train_models.py`)

### Performance Optimization

- Use GPU acceleration for XGBoost (if available)
- Reduce feature selection count for faster training
- Use smaller hyperparameter search spaces

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## 👥 Team Name : pheonix


##  Acknowledgments

- CIC Research for providing the CICIDS-2017 dataset
- ISCX for providing the VPN-NonVPN dataset
- Scikit-learn and XGBoost communities for excellent ML libraries



**Note**: This repository contains the core implementation. For complete datasets, reports, and demonstrations, please refer to the Google Drive link provided above.
