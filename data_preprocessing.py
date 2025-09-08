"""
Data preprocessing utilities for Skin Allergy Risk Prediction
Handles data loading, cleaning, encoding, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import logging
from typing import Tuple, Dict, List
import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL),
                   format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Comprehensive data preprocessor for skin allergy prediction dataset
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_names = []
        self.target_encoder = LabelEncoder()
        self.is_fitted = False
        
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """Load the dataset from CSV file"""
        if filepath is None:
            filepath = config.DATASET_FILE
            
        try:
            logger.info(f"Loading dataset from {filepath}")
            df = pd.read_csv(filepath)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataset"""
        logger.info("Starting data cleaning...")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Check for missing values
        missing_values = df_clean.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Found {missing_values.sum()} missing values")
            logger.info("Missing values by column:")
            for col, count in missing_values[missing_values > 0].items():
                logger.info(f"  {col}: {count}")
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # Validate data ranges
        df_clean = self._validate_data_ranges(df_clean)
        
        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using appropriate strategies"""
        df_filled = df.copy()
        
        # Fill numerical features with median
        for col in config.NUMERICAL_FEATURES:
            if col in df_filled.columns and df_filled[col].isnull().any():
                median_val = df_filled[col].median()
                df_filled[col].fillna(median_val, inplace=True)
                logger.info(f"Filled missing values in {col} with median: {median_val}")
        
        # Fill categorical features with mode
        for col in config.CATEGORICAL_FEATURES:
            if col in df_filled.columns and df_filled[col].isnull().any():
                mode_val = df_filled[col].mode()[0]
                df_filled[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled missing values in {col} with mode: {mode_val}")
        
        # Fill binary features with 0 (most conservative approach)
        for col in config.BINARY_FEATURES:
            if col in df_filled.columns and df_filled[col].isnull().any():
                df_filled[col].fillna(0, inplace=True)
                logger.info(f"Filled missing values in {col} with 0")
        
        return df_filled
    
    def _validate_data_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix data ranges"""
        df_validated = df.copy()
        
        # Age validation
        if 'age' in df_validated.columns:
            df_validated['age'] = np.clip(df_validated['age'], 0, 120)
        
        # Percentage validations
        percentage_cols = ['humidity_percent']
        for col in percentage_cols:
            if col in df_validated.columns:
                df_validated[col] = np.clip(df_validated[col], 0, 100)
        
        # Score validations (1-10 scale)
        score_cols = ['stress_level', 'sleep_quality_score']
        for col in score_cols:
            if col in df_validated.columns:
                df_validated[col] = np.clip(df_validated[col], 1, 10)
        
        # UV Index validation (0-12 scale)
        if 'uv_index' in df_validated.columns:
            df_validated['uv_index'] = np.clip(df_validated['uv_index'], 0, 12)
        
        # Exercise frequency validation (0-7 days per week)
        if 'exercise_frequency_per_week' in df_validated.columns:
            df_validated['exercise_frequency_per_week'] = np.clip(
                df_validated['exercise_frequency_per_week'], 0, 7
            )
        
        return df_validated
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from existing ones"""
        logger.info("Starting feature engineering...")
        
        df_engineered = df.copy()
        
        # Age groups
        if 'age' in df_engineered.columns:
            df_engineered['age_group'] = pd.cut(
                df_engineered['age'],
                bins=[0, 18, 35, 55, 120],
                labels=['Child', 'Young_Adult', 'Adult', 'Senior']
            )
        
        # BMI category (if height and weight were available, but we'll simulate)
        # Environmental risk score
        env_cols = ['humidity_percent', 'temperature_celsius', 'air_quality_index', 
                   'uv_index', 'pollen_count']
        if all(col in df_engineered.columns for col in env_cols):
            # Normalize environmental factors and create composite score
            env_risk = 0
            if 'humidity_percent' in df_engineered.columns:
                # Risk increases at humidity extremes
                humidity_risk = np.abs(df_engineered['humidity_percent'] - 50) / 50
                env_risk += humidity_risk * 0.2
            
            if 'air_quality_index' in df_engineered.columns:
                # Higher AQI = higher risk
                aqi_risk = df_engineered['air_quality_index'] / 500
                env_risk += aqi_risk * 0.3
            
            if 'uv_index' in df_engineered.columns:
                # Higher UV = higher risk
                uv_risk = df_engineered['uv_index'] / 12
                env_risk += uv_risk * 0.2
            
            if 'pollen_count' in df_engineered.columns:
                # Higher pollen = higher risk
                pollen_risk = np.minimum(df_engineered['pollen_count'] / 200, 1)
                env_risk += pollen_risk * 0.3
            
            df_engineered['environmental_risk_score'] = env_risk
        
        # Lifestyle risk score
        lifestyle_risk = 0
        if 'stress_level' in df_engineered.columns:
            lifestyle_risk += (df_engineered['stress_level'] - 1) / 9 * 0.4
        
        if 'sleep_quality_score' in df_engineered.columns:
            # Poor sleep = higher risk
            lifestyle_risk += (10 - df_engineered['sleep_quality_score']) / 9 * 0.3
        
        if 'exercise_frequency_per_week' in df_engineered.columns:
            # Sedentary lifestyle = higher risk
            lifestyle_risk += (7 - df_engineered['exercise_frequency_per_week']) / 7 * 0.3
        
        df_engineered['lifestyle_risk_score'] = lifestyle_risk
        
        # Product exposure risk score
        product_cols = ['cosmetic_usage_frequency', 'skincare_routine_frequency',
                       'hair_product_usage', 'fragrance_usage_frequency']
        if all(col in df_engineered.columns for col in product_cols):
            product_risk = (
                df_engineered['cosmetic_usage_frequency'] * 0.3 +
                df_engineered['skincare_routine_frequency'] * 0.2 +
                df_engineered['hair_product_usage'] * 0.2 +
                df_engineered['fragrance_usage_frequency'] * 0.3
            ) / 10  # Normalize to 0-1 scale
            
            df_engineered['product_exposure_score'] = product_risk
        
        # Medical history composite score
        medical_cols = ['family_history_allergies', 'previous_allergic_reactions',
                       'asthma', 'eczema', 'autoimmune_conditions']
        if all(col in df_engineered.columns for col in medical_cols):
            medical_score = (
                df_engineered['family_history_allergies'] * 0.2 +
                df_engineered['previous_allergic_reactions'] * 0.3 +
                df_engineered['asthma'] * 0.2 +
                df_engineered['eczema'] * 0.2 +
                df_engineered['autoimmune_conditions'] * 0.1
            )
            df_engineered['medical_history_score'] = medical_score
        
        # Seasonal risk adjustment
        if 'season' in df_engineered.columns:
            season_risk_map = {'Spring': 1.0, 'Summer': 0.7, 'Fall': 0.5, 'Winter': 0.3}
            df_engineered['seasonal_risk_factor'] = df_engineered['season'].map(season_risk_map)
        
        logger.info(f"Feature engineering completed. New shape: {df_engineered.shape}")
        return df_engineered
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        # Encode categorical features
        for col in config.CATEGORICAL_FEATURES + ['age_group']:
            if col in df_encoded.columns:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        unique_values = set(df_encoded[col].astype(str))
                        known_values = set(self.label_encoders[col].classes_)
                        unseen_values = unique_values - known_values
                        
                        if unseen_values:
                            logger.warning(f"Unseen categories in {col}: {unseen_values}")
                            # Replace unseen values with the most frequent class
                            most_frequent = self.label_encoders[col].classes_[0]
                            df_encoded[col] = df_encoded[col].astype(str).replace(
                                list(unseen_values), most_frequent
                            )
                        
                        df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def prepare_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target variable"""
        logger.info("Preparing features and target...")
        
        # Define feature columns (exclude target and ID columns)
        exclude_cols = [config.TARGET_COLUMN, config.RISK_SCORE_COLUMN]
        
        # Remove any ID-like columns
        exclude_cols.extend([col for col in df.columns if 'id' in col.lower()])
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[config.TARGET_COLUMN].copy()
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Features prepared: {len(feature_cols)} features")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        return X, y
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        logger.info("Scaling numerical features...")
        
        X_scaled = X.copy()
        
        # Identify numerical columns to scale
        numerical_cols = []
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64'] and col not in config.BINARY_FEATURES:
                numerical_cols.append(col)
        
        if numerical_cols:
            if fit:
                X_scaled[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
            else:
                X_scaled[numerical_cols] = self.scaler.transform(X[numerical_cols])
            
            logger.info(f"Scaled {len(numerical_cols)} numerical features")
        
        return X_scaled
    
    def encode_target(self, y: pd.Series, fit: bool = True) -> np.ndarray:
        """Encode target variable"""
        if fit:
            y_encoded = self.target_encoder.fit_transform(y)
            logger.info(f"Target classes: {self.target_encoder.classes_}")
        else:
            y_encoded = self.target_encoder.transform(y)
        
        return y_encoded
    
    def split_data(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Split data into training and testing sets"""
        logger.info("Splitting data into train/test sets...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Testing set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Complete preprocessing pipeline for training data"""
        logger.info("Starting complete preprocessing pipeline...")
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Feature engineering
        df_features = self.engineer_features(df_clean)
        
        # Encode categorical features
        df_encoded = self.encode_features(df_features, fit=True)
        
        # Prepare features and target
        X, y = self.prepare_features_target(df_encoded)
        
        # Scale features
        X_scaled = self.scale_features(X, fit=True)
        
        # Encode target
        y_encoded = self.encode_target(y, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X_scaled, y_encoded)
        
        self.is_fitted = True
        logger.info("Preprocessing pipeline completed successfully!")
        
        return X_train, X_test, y_train, y_test
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first!")
        
        logger.info("Transforming new data...")
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Feature engineering
        df_features = self.engineer_features(df_clean)
        
        # Encode categorical features
        df_encoded = self.encode_features(df_features, fit=False)
        
        # Prepare features
        X, _ = self.prepare_features_target(df_encoded)
        
        # Scale features
        X_scaled = self.scale_features(X, fit=False)
        
        logger.info("Data transformation completed!")
        return X_scaled
    
    def transform_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data for prediction (when target column doesn't exist)"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first!")
        
        logger.info("Transforming prediction data...")
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Feature engineering
        df_features = self.engineer_features(df_clean)
        
        # Encode categorical features
        df_encoded = self.encode_features(df_features, fit=False)
        
        # Prepare features without target column
        exclude_cols = [config.TARGET_COLUMN, config.RISK_SCORE_COLUMN]
        exclude_cols.extend([col for col in df_encoded.columns if 'id' in col.lower()])
        feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
        X = df_encoded[feature_cols].copy()
        
        # Scale features
        X_scaled = self.scale_features(X, fit=False)
        
        logger.info("Prediction data transformation completed!")
        return X_scaled
    
    def save_preprocessor(self, filepath: str = None):
        """Save the fitted preprocessor"""
        if filepath is None:
            filepath = f"{config.MODELS_DIR}/{config.PREPROCESSOR_FILENAME}"
        
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'target_encoder': self.target_encoder,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str = None):
        """Load a fitted preprocessor"""
        if filepath is None:
            filepath = f"{config.MODELS_DIR}/{config.PREPROCESSOR_FILENAME}"
        
        preprocessor_data = joblib.load(filepath)
        
        self.label_encoders = preprocessor_data['label_encoders']
        self.scaler = preprocessor_data['scaler']
        self.target_encoder = preprocessor_data['target_encoder']
        self.feature_names = preprocessor_data['feature_names']
        self.is_fitted = preprocessor_data['is_fitted']
        
        logger.info(f"Preprocessor loaded from {filepath}")


def get_feature_info() -> Dict[str, List[str]]:
    """Get information about different types of features"""
    return {
        'categorical': config.CATEGORICAL_FEATURES,
        'numerical': config.NUMERICAL_FEATURES,
        'binary': config.BINARY_FEATURES,
        'target': config.TARGET_COLUMN
    }
