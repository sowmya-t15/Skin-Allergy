"""
XGBoost Model implementation for Skin Allergy Risk Prediction
Handles model training, evaluation, and prediction
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, log_loss)
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL),
                   format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class XGBoostSkinAllergyModel:
    """
    XGBoost model for skin allergy risk prediction with comprehensive evaluation
    """
    
    def __init__(self, params: Dict = None):
        """Initialize the XGBoost model with specified parameters"""
        self.params = params if params else config.XGBOOST_PARAMS.copy()
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.feature_importance = {}
        self.training_history = {}
        
        logger.info("XGBoost Skin Allergy Model initialized")
        logger.info(f"Model parameters: {self.params}")
    
    def create_model(self) -> xgb.XGBClassifier:
        """Create XGBoost classifier with specified parameters"""
        self.model = xgb.XGBClassifier(**self.params)
        logger.info("XGBoost model created successfully")
        return self.model
    
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray,
              X_val: pd.DataFrame = None, y_val: np.ndarray = None,
              early_stopping_rounds: int = 50) -> Dict[str, Any]:
        """
        Train the XGBoost model with optional validation and early stopping
        """
        logger.info("Starting model training...")
        
        if self.model is None:
            self.create_model()
        
        self.feature_names = list(X_train.columns)
        
        # Prepare evaluation set for early stopping
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        else:
            eval_set = [(X_train, y_train)]
        
        # Train the model with compatibility for different XGBoost versions
        try:
            # Try new XGBoost API first
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        except TypeError as e:
            # Fallback for older XGBoost versions that don't support early_stopping_rounds
            logger.warning(f"Early stopping not supported in this XGBoost version: {str(e)}")
            logger.info("Training without early stopping...")
            self.model.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            # Last resort: basic training
            logger.info("Attempting basic training without validation...")
            self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Store feature importance
        self._calculate_feature_importance()
        
        # Get training metrics
        training_metrics = self._get_training_metrics()
        
        logger.info("Model training completed successfully!")
        if hasattr(self.model, 'best_iteration'):
            logger.info(f"Best iteration: {self.model.best_iteration}")
        if hasattr(self.model, 'best_score'):
            logger.info(f"Best score: {self.model.best_score:.4f}")
        
        return training_metrics
    
    def _calculate_feature_importance(self):
        """Calculate and store feature importance"""
        if not self.is_trained:
            logger.warning("Model not trained yet!")
            return
        
        # Get different types of feature importance
        importance_types = ['weight', 'gain', 'cover']
        
        for imp_type in importance_types:
            importance = self.model.get_booster().get_score(importance_type=imp_type)
            
            # Convert to feature names if using default feature names
            if self.feature_names:
                importance_named = {}
                for i, feature_name in enumerate(self.feature_names):
                    feature_key = f'f{i}'
                    if feature_key in importance:
                        importance_named[feature_name] = importance[feature_key]
                self.feature_importance[imp_type] = importance_named
            else:
                self.feature_importance[imp_type] = importance
        
        logger.info("Feature importance calculated")
    
    def _get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics and history"""
        if not self.is_trained:
            return {}
        
        metrics = {
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names
        }
        
        # Get evaluation results if available (for newer XGBoost versions)
        if hasattr(self.model, 'best_iteration'):
            metrics['best_iteration'] = self.model.best_iteration
        
        if hasattr(self.model, 'best_score'):
            metrics['best_score'] = self.model.best_score
            
        if hasattr(self.model, 'evals_result_'):
            metrics['eval_results'] = self.model.evals_result_
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        logger.info(f"Making predictions for {len(X)} samples...")
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        logger.info(f"Getting prediction probabilities for {len(X)} samples...")
        probabilities = self.model.predict_proba(X)
        
        return probabilities
    
    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive evaluation of the model
        """
        logger.info("Starting model evaluation...")
        
        # Get predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro')
        metrics['precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
        metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro')
        metrics['recall_weighted'] = recall_score(y_test, y_pred, average='weighted')
        metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
        
        # Multi-class ROC AUC
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(y_test, y_pred_proba, 
                                                  multi_class='ovr', average='weighted')
            metrics['roc_auc_ovo'] = roc_auc_score(y_test, y_pred_proba, 
                                                  multi_class='ovo', average='weighted')
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {str(e)}")
            metrics['roc_auc_ovr'] = 0.0
            metrics['roc_auc_ovo'] = 0.0
        
        # Log loss
        metrics['log_loss'] = log_loss(y_test, y_pred_proba)
        
        # Per-class metrics
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Add per-class metrics to main metrics dict
        for class_label, class_metrics in class_report.items():
            if isinstance(class_metrics, dict) and class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                for metric_name, value in class_metrics.items():
                    metrics[f'class_{class_label}_{metric_name}'] = value
        
        logger.info("Model evaluation completed!")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
        logger.info(f"ROC AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
        
        return metrics, y_pred, y_pred_proba
    
    def cross_validate(self, X: pd.DataFrame, y: np.ndarray, 
                      cv_folds: int = None) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation
        """
        if cv_folds is None:
            cv_folds = config.CV_FOLDS
        
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        if self.model is None:
            self.create_model()
        
        # Stratified K-Fold for balanced splits
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                             random_state=config.RANDOM_STATE)
        
        cv_results = {}
        
        # Perform cross-validation for different metrics
        for metric in config.SCORING_METRICS:
            try:
                scores = cross_val_score(self.model, X, y, cv=skf, 
                                       scoring=metric, n_jobs=-1)
                cv_results[metric] = scores
                
                logger.info(f"{metric.upper()} CV Scores: "
                           f"{scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            except Exception as e:
                logger.warning(f"Could not calculate {metric}: {str(e)}")
                cv_results[metric] = np.array([0.0] * cv_folds)
        
        return cv_results
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: np.ndarray,
                             param_grid: Dict = None, cv_folds: int = 3) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV
        """
        logger.info("Starting hyperparameter tuning...")
        
        if param_grid is None:
            param_grid = config.PARAM_GRID
        
        # Create a fresh model for tuning
        base_params = config.XGBOOST_PARAMS.copy()
        # Remove problematic parameters for older versions
        if 'eval_metric' in base_params:
            del base_params['eval_metric']
        
        base_model = xgb.XGBClassifier(**base_params)
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                              random_state=config.RANDOM_STATE),
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        try:
            grid_search.fit(X_train, y_train)
            
            # Update model with best parameters
            self.params.update(grid_search.best_params_)
            self.model = grid_search.best_estimator_
            self.is_trained = True
            
            # Store feature names
            self.feature_names = list(X_train.columns)
            
            # Store feature importance for the best model
            self._calculate_feature_importance()
            
            tuning_results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            logger.info("Hyperparameter tuning completed!")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {str(e)}")
            logger.info("Falling back to default parameters...")
            
            # Fallback: use default parameters
            self.create_model()
            tuning_results = {
                'best_params': self.params,
                'best_score': 0.0,
                'cv_results': {},
                'error': str(e)
            }
        
        return tuning_results
    
    def get_feature_importance(self, importance_type: str = 'gain', 
                             top_n: int = None) -> Dict[str, float]:
        """
        Get feature importance scores
        """
        if not self.is_trained:
            logger.warning("Model not trained yet!")
            return {}
        
        if top_n is None:
            top_n = config.TOP_FEATURES_COUNT
        
        if importance_type not in self.feature_importance:
            logger.warning(f"Importance type '{importance_type}' not available")
            return {}
        
        importance = self.feature_importance[importance_type]
        
        # Sort by importance and get top N
        sorted_importance = dict(sorted(importance.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        if top_n:
            sorted_importance = dict(list(sorted_importance.items())[:top_n])
        
        return sorted_importance
    
    def plot_feature_importance(self, importance_type: str = 'gain', 
                               top_n: int = None, save_path: str = None):
        """
        Plot feature importance
        """
        if top_n is None:
            top_n = config.TOP_FEATURES_COUNT
        
        importance = self.get_feature_importance(importance_type, top_n)
        
        if not importance:
            logger.warning("No feature importance data available")
            return
        
        # Create plot
        plt.figure(figsize=config.FIGURE_SIZE)
        
        features = list(importance.keys())
        scores = list(importance.values())
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        
        plt.barh(y_pos, scores, color='steelblue', alpha=0.7)
        plt.yticks(y_pos, features)
        plt.xlabel(f'Feature Importance ({importance_type})')
        plt.title(f'Top {len(features)} Most Important Features')
        plt.gca().invert_yaxis()  # Highest importance at top
        
        # Add value labels on bars
        for i, score in enumerate(scores):
            plt.text(score + max(scores) * 0.01, i, f'{score:.3f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str = None):
        """Save the trained model"""
        if not self.is_trained:
            logger.warning("Model not trained yet!")
            return
        
        if filepath is None:
            filepath = f"{config.MODELS_DIR}/{config.MODEL_FILENAME}"
        
        model_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = None):
        """Load a trained model"""
        if filepath is None:
            filepath = f"{config.MODELS_DIR}/{config.MODEL_FILENAME}"
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.training_history = model_data['training_history']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        info = {
            'is_trained': self.is_trained,
            'parameters': self.params,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names
        }
        
        if self.is_trained and self.model:
            if hasattr(self.model, 'best_iteration'):
                info['best_iteration'] = self.model.best_iteration
            if hasattr(self.model, 'best_score'):
                info['best_score'] = self.model.best_score
            if hasattr(self.model, 'n_estimators'):
                info['n_estimators'] = self.model.n_estimators
        
        return info


def create_ensemble_model(models: List[XGBoostSkinAllergyModel], 
                         weights: List[float] = None) -> 'EnsembleModel':
    """
    Create an ensemble of XGBoost models
    """
    return EnsembleModel(models, weights)


class EnsembleModel:
    """
    Simple ensemble model that combines multiple XGBoost models
    """
    
    def __init__(self, models: List[XGBoostSkinAllergyModel], 
                 weights: List[float] = None):
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
        self.n_models = len(models)
        
        # Validate weights
        if len(self.weights) != self.n_models:
            raise ValueError("Number of weights must match number of models")
        
        if abs(sum(self.weights) - 1.0) > 1e-6:
            logger.warning("Weights don't sum to 1.0, normalizing...")
            total_weight = sum(self.weights)
            self.weights = [w/total_weight for w in self.weights]
        
        logger.info(f"Ensemble model created with {self.n_models} models")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble prediction probabilities"""
        predictions = []
        
        for model in self.models:
            pred_proba = model.predict_proba(X)
            predictions.append(pred_proba)
        
        # Weighted average of predictions
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return ensemble_pred
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble predictions"""
        pred_proba = self.predict_proba(X)
        return np.argmax(pred_proba, axis=1)
