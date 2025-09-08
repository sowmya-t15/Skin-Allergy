# 🧴 Skin Allergy Risk Prediction System

A comprehensive machine learning system for predicting skin allergy risk using XGBoost, built with Python and Flask. This project includes synthetic dataset generation, model training, evaluation, and a web API for real-time predictions.

## 🌟 Features

- **Synthetic Dataset Generation**: Creates realistic skin allergy data with proper medical correlations
- **XGBoost Model**: High-performance gradient boosting for accurate predictions
- **Comprehensive Evaluation**: Detailed metrics and visualizations
- **Web API**: Flask-based REST API with interactive web interface
- **Command-Line Interface**: Easy-to-use CLI for all operations
- **Modular Design**: Well-structured codebase for easy maintenance and extension

## 📊 Dataset Features

The system considers multiple factors for allergy risk prediction:

### Personal Information
- Age, Gender, Skin type (Fitzpatrick scale)
- Medical history (allergies, asthma, eczema)
- Family history of allergies

### Environmental Factors
- Season, Humidity, Temperature
- Air quality index, UV index, Pollen count

### Lifestyle Factors
- Stress level, Sleep quality
- Exercise frequency, Diet type
- Smoking status, Alcohol consumption

### Product Exposure
- Cosmetic usage frequency
- Skincare routine frequency
- Hair product and fragrance usage
- Chemical exposure (household/occupational)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd "major project"

# Activate virtual environment
.\test\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python main.py generate-data --samples 5000
```

### 3. Train Model

```bash
python main.py train
```

### 4. Start Web API

```bash
python main.py api
```

Then open http://127.0.0.1:5000 in your browser to use the web interface.

## 📁 Project Structure

```
major project/
├── main.py                          # Main CLI entry point
├── config.py                        # Configuration settings
├── data_preprocessing.py             # Data preprocessing utilities
├── model.py                         # XGBoost model implementation
├── train_test.py                    # Training and testing pipelines
├── evaluation.py                    # Model evaluation utilities
├── api.py                          # Flask web API
├── generate_skin_allergy_dataset.py # Dataset generation
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
├── data/                           # Generated datasets
├── models/                         # Trained models
├── results/                        # Training results and plots
└── logs/                          # Log files
```

## 🔧 Usage

### Command Line Interface

```bash
# Show project information
python main.py info

# Generate synthetic dataset
python main.py generate-data --samples 10000

# Train model with hyperparameter tuning
python main.py train

# Train model without tuning (faster)
python main.py train --no-tuning

# Test trained model
python main.py test --test-data path/to/test.csv

# Start web API server
python main.py api --host 0.0.0.0 --port 8080

# Get help for specific commands
python main.py train --help
```

### Python API Usage

```python
from data_preprocessing import DataPreprocessor
from model import XGBoostSkinAllergyModel
import pandas as pd

# Load and preprocess data
preprocessor = DataPreprocessor()
df = preprocessor.load_data('skin_allergy_dataset.csv')
X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

# Train model
model = XGBoostSkinAllergyModel()
model.train(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Evaluate model
metrics, y_pred, y_pred_proba = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Web API Endpoints

- `GET /` - Web interface for interactive predictions
- `POST /predict` - Single prediction endpoint
- `POST /batch_predict` - Batch prediction endpoint
- `GET /health` - Health check endpoint
- `GET /model_info` - Model information and feature importance

#### Example API Request

```python
import requests

# Single prediction
data = {
    "age": 30,
    "gender": "Female",
    "skin_type": "Type_III",
    "stress_level": 7,
    "previous_allergic_reactions": 1,
    "cosmetic_usage_frequency": 8
}

response = requests.post('http://127.0.0.1:5000/predict', json=data)
result = response.json()
print(f"Risk Level: {result['risk_level']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## 📈 Model Performance

The XGBoost model typically achieves:
- **Accuracy**: 85-90%
- **F1 Score**: 85-88%
- **ROC AUC**: 88-92%

Performance metrics include:
- Basic classification metrics (accuracy, precision, recall, F1)
- Multi-class ROC AUC (One-vs-Rest and One-vs-One)
- Risk-weighted accuracy (penalizes larger misclassifications)
- Confidence and calibration metrics
- Per-class performance analysis

## 🎯 Risk Categories

- **Low Risk (0-40)**: Minimal likelihood of allergic reactions
- **Medium Risk (40-70)**: Moderate risk, preventive measures recommended
- **High Risk (70-100)**: High likelihood, consultation with dermatologist advised

## 🔍 Feature Importance

The model automatically identifies the most important factors for allergy prediction:

1. **Previous allergic reactions** (strongest predictor)
2. **Family history of allergies**
3. **Eczema/atopic dermatitis**
4. **Known primary allergen**
5. **Stress level**
6. **Cosmetic usage frequency**
7. **Age group**
8. **Seasonal factors**
9. **Environmental conditions**
10. **Product exposure**

## 📊 Visualizations

The system generates comprehensive visualizations:
- Feature importance plots
- Confusion matrices
- ROC and Precision-Recall curves
- Training curves
- Prediction confidence distributions
- Calibration curves

## ⚙️ Configuration

Key parameters can be modified in `config.py`:

```python
# Model parameters
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    # ... other parameters
}

# Data split ratios
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15

# Feature importance settings
TOP_FEATURES_COUNT = 15
```

## 🚨 Important Notes

1. **Medical Disclaimer**: This system is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

2. **Synthetic Data**: The default dataset is synthetically generated with realistic correlations. For production use, replace with real medical data following appropriate ethical and privacy guidelines.

3. **Model Limitations**: The model's predictions are based on the patterns learned from training data and may not capture all real-world complexities.

## 🔮 Future Enhancements

- Integration with real medical databases
- Advanced feature engineering (genetic markers, environmental sensors)
- Deep learning models for complex pattern recognition
- Mobile app development
- Integration with wearable devices
- Multi-language support for global deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Created for skin allergy risk prediction research and education.

## 📞 Support

For questions or issues:
- Check the documentation in each Python file
- Review the example usage in `main.py`
- Examine the configuration options in `config.py`

---

**Remember**: Always consult with healthcare professionals for actual medical decisions. This tool is designed for educational and research purposes.
