"""
Configuration file for Skin Allergy Risk Prediction Project
Contains all model parameters, file paths, and feature definitions
"""

import os

# Project Configuration
PROJECT_NAME = "Skin Allergy Risk Prediction"
VERSION = "1.0.0"
RANDOM_STATE = 42

# File Paths
DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"
LOGS_DIR = "logs"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data Configuration
DATASET_FILE = os.path.join(DATA_DIR, "skin_allergy_dataset_20250813_193030.csv")
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15

# Feature Configuration
CATEGORICAL_FEATURES = [
    'gender', 'skin_type', 'occupation', 'known_primary_allergen',
    'season', 'diet_type', 'smoking_status'
]

NUMERICAL_FEATURES = [
    'age', 'humidity_percent', 'temperature_celsius', 'air_quality_index',
    'uv_index', 'pollen_count', 'stress_level', 'sleep_quality_score',
    'exercise_frequency_per_week', 'alcohol_consumption_per_week',
    'cosmetic_usage_frequency', 'skincare_routine_frequency',
    'hair_product_usage', 'fragrance_usage_frequency',
    'household_chemical_exposure', 'occupational_chemical_exposure'
]

BINARY_FEATURES = [
    'family_history_allergies', 'previous_allergic_reactions',
    'asthma', 'eczema', 'autoimmune_conditions', 'new_products_tried_recently'
]

TARGET_COLUMN = 'allergy_risk_level'
RISK_SCORE_COLUMN = 'allergy_risk_score'

# XGBoost Model Configuration
XGBOOST_PARAMS = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': 3,
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': 1
}

# Hyperparameter Tuning Grid
PARAM_GRID = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.15],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Cross-validation Configuration
CV_FOLDS = 5
SCORING_METRICS = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']

# Model Evaluation Thresholds
RISK_THRESHOLDS = {
    'Low': (0, 40),
    'Medium': (40, 70),
    'High': (70, 100)
}

# Feature Importance Configuration
TOP_FEATURES_COUNT = 15
FEATURE_IMPORTANCE_THRESHOLD = 0.01

# Visualization Configuration
FIGURE_SIZE = (12, 8)
DPI = 300
COLOR_PALETTE = ['#2E8B57', '#FFD700', '#FF6347']  # Green, Yellow, Red for Low, Medium, High
STYLE = 'whitegrid'

# Model Persistence
MODEL_FILENAME = 'xgboost_skin_allergy_model.pkl'
PREPROCESSOR_FILENAME = 'data_preprocessor.pkl'
FEATURE_NAMES_FILENAME = 'feature_names.pkl'

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# API Configuration (for future deployment)
API_HOST = '0.0.0.0'
API_PORT = 5000
API_DEBUG = True

# Performance Monitoring
PERFORMANCE_THRESHOLD = 0.85  # Minimum acceptable accuracy
DRIFT_THRESHOLD = 0.1  # Maximum acceptable performance drift
