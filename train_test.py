"""
Training and Testing Pipeline for Skin Allergy Risk Prediction
Orchestrates the complete machine learning workflow
"""

import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import time

import config
from data_preprocessing import DataPreprocessor
from model import XGBoostSkinAllergyModel
from evaluation import ModelEvaluator

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL),
                   format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """
    Complete training pipeline for skin allergy risk prediction
    """
    
    def __init__(self, data_file: str = None):
        """Initialize the training pipeline"""
        self.data_file = data_file or config.DATASET_FILE
        self.preprocessor = DataPreprocessor()
        self.model = XGBoostSkinAllergyModel()
        self.evaluator = ModelEvaluator()
        
        # Training results storage
        self.results = {}
        self.training_time = None
        
        logger.info("Training Pipeline initialized")
    
    def load_data(self) -> pd.DataFrame:
        """Load the dataset"""
        logger.info("=" * 50)
        logger.info("STEP 1: LOADING DATA")
        logger.info("=" * 50)
        
        df = self.preprocessor.load_data(self.data_file)
        
        # Log basic dataset info
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Features: {df.columns.tolist()}")
        logger.info("\nTarget variable distribution:")
        logger.info(df[config.TARGET_COLUMN].value_counts())
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> tuple:
        """Preprocess the data"""
        logger.info("=" * 50)
        logger.info("STEP 2: DATA PREPROCESSING")
        logger.info("=" * 50)
        
        # Complete preprocessing pipeline
        X_train, X_test, y_train, y_test = self.preprocessor.fit_transform(df)
        
        # Create validation set from training data
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train,
            test_size=config.VALIDATION_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y_train
        )
        
        logger.info(f"Training set: {X_train_split.shape}")
        logger.info(f"Validation set: {X_val.shape}")
        logger.info(f"Test set: {X_test.shape}")
        
        return X_train_split, X_val, X_test, y_train_split, y_val, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: np.ndarray,
                   X_val: pd.DataFrame, y_val: np.ndarray,
                   perform_tuning: bool = True) -> dict:
        """Train the XGBoost model"""
        logger.info("=" * 50)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("=" * 50)
        
        start_time = datetime.now()
        
        if perform_tuning:
            logger.info("🔍 Performing hyperparameter tuning...")
            print("🔍 Starting hyperparameter tuning (this may take a while)...")
            
            # Show progress for hyperparameter tuning
            with tqdm(total=100, desc="Hyperparameter Tuning", ncols=100) as pbar:
                tuning_results = self.model.hyperparameter_tuning(X_train, y_train)
                pbar.update(100)
            
            self.results['hyperparameter_tuning'] = tuning_results
            logger.info("✅ Hyperparameter tuning completed!")
            print("✅ Hyperparameter tuning completed!")
        
        # Train the model
        logger.info("🚀 Training final model...")
        print("🚀 Training final model with best parameters...")
        
        # Show training progress
        with tqdm(total=100, desc="Model Training", ncols=100) as pbar:
            training_metrics = self.model.train(X_train, y_train, X_val, y_val)
            pbar.update(100)
        
        self.training_time = datetime.now() - start_time
        logger.info(f"⏱️  Training completed in: {self.training_time}")
        print(f"⏱️  Training completed in: {self.training_time}")
        
        self.results['training_metrics'] = training_metrics
        return training_metrics
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
        """Evaluate the trained model"""
        logger.info("=" * 50)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("=" * 50)
        
        # Evaluate on test set
        test_metrics, y_pred, y_pred_proba = self.model.evaluate(X_test, y_test)
        
        # Generate comprehensive evaluation report
        evaluation_results = self.evaluator.generate_evaluation_report(
            y_test, y_pred, y_pred_proba,
            self.preprocessor.target_encoder.classes_
        )
        
        self.results['test_metrics'] = test_metrics
        self.results['evaluation_results'] = evaluation_results
        self.results['predictions'] = {
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist()
        }
        
        return evaluation_results
    
    def perform_cross_validation(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        """Perform cross-validation"""
        logger.info("=" * 50)
        logger.info("STEP 5: CROSS-VALIDATION")
        logger.info("=" * 50)
        
        cv_results = self.model.cross_validate(X, y)
        self.results['cross_validation'] = cv_results
        
        # Log CV results
        for metric, scores in cv_results.items():
            logger.info(f"{metric.upper()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results
    
    def analyze_feature_importance(self) -> dict:
        """Analyze feature importance"""
        logger.info("=" * 50)
        logger.info("STEP 6: FEATURE IMPORTANCE ANALYSIS")
        logger.info("=" * 50)
        
        # Get feature importance for different types
        importance_results = {}
        importance_types = ['weight', 'gain', 'cover']
        
        for imp_type in importance_types:
            importance = self.model.get_feature_importance(imp_type)
            importance_results[imp_type] = importance
            
            logger.info(f"\nTop 10 features by {imp_type}:")
            for i, (feature, score) in enumerate(list(importance.items())[:10], 1):
                logger.info(f"  {i:2d}. {feature:30s}: {score:.4f}")
        
        self.results['feature_importance'] = importance_results
        return importance_results
    
    def save_results(self) -> str:
        """Save all results to files"""
        logger.info("=" * 50)
        logger.info("STEP 7: SAVING RESULTS")
        logger.info("=" * 50)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save preprocessor
        self.preprocessor.save_preprocessor()
        
        # Save model
        self.model.save_model()
        
        # Save results as JSON
        results_file = f"{config.RESULTS_DIR}/training_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._prepare_results_for_json()
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Save detailed metrics
        metrics_file = f"{config.RESULTS_DIR}/detailed_metrics_{timestamp}.txt"
        self._save_detailed_metrics(metrics_file)
        
        return results_file
    
    def _prepare_results_for_json(self) -> dict:
        """Prepare results for JSON serialization"""
        def convert_value(value):
            """Recursively convert numpy types to Python types"""
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                return int(value)
            elif isinstance(value, (np.floating, np.float16, np.float32, np.float64)):
                return float(value)
            elif isinstance(value, (np.bool_, bool)):
                return bool(value)
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [convert_value(item) for item in value]
            else:
                return value
        
        json_results = {}
        
        for key, value in self.results.items():
            json_results[key] = convert_value(value)
        
        # Add metadata
        json_results['metadata'] = {
            'training_date': datetime.now().isoformat(),
            'training_time': str(self.training_time),
            'dataset_file': self.data_file,
            'model_type': 'XGBoost',
            'config_version': config.VERSION
        }
        
        return json_results
    
    def _save_detailed_metrics(self, filepath: str):
        """Save detailed metrics to text file"""
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SKIN ALLERGY RISK PREDICTION - DETAILED METRICS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training Time: {self.training_time}\n")
            f.write(f"Dataset: {self.data_file}\n\n")
            
            # Model parameters
            f.write("MODEL PARAMETERS:\n")
            f.write("-" * 40 + "\n")
            for param, value in self.model.params.items():
                f.write(f"{param:20s}: {value}\n")
            f.write("\n")
            
            # Test metrics
            if 'test_metrics' in self.results:
                f.write("TEST SET METRICS:\n")
                f.write("-" * 40 + "\n")
                for metric, value in self.results['test_metrics'].items():
                    if isinstance(value, float):
                        f.write(f"{metric:25s}: {value:.4f}\n")
                    else:
                        f.write(f"{metric:25s}: {value}\n")
                f.write("\n")
            
            # Cross-validation results
            if 'cross_validation' in self.results:
                f.write("CROSS-VALIDATION RESULTS:\n")
                f.write("-" * 40 + "\n")
                for metric, scores in self.results['cross_validation'].items():
                    if isinstance(scores, np.ndarray):
                        f.write(f"{metric:20s}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})\n")
                f.write("\n")
            
            # Feature importance
            if 'feature_importance' in self.results:
                f.write("FEATURE IMPORTANCE (Top 15):\n")
                f.write("-" * 40 + "\n")
                importance = self.results['feature_importance'].get('gain', {})
                for i, (feature, score) in enumerate(list(importance.items())[:15], 1):
                    f.write(f"{i:2d}. {feature:30s}: {score:.4f}\n")
        
        logger.info(f"Detailed metrics saved to {filepath}")
    
    def generate_visualizations(self):
        """Generate and save visualizations"""
        logger.info("=" * 50)
        logger.info("STEP 8: GENERATING VISUALIZATIONS")
        logger.info("=" * 50)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot feature importance
        importance_plot = f"{config.RESULTS_DIR}/feature_importance_{timestamp}.png"
        self.model.plot_feature_importance(save_path=importance_plot)
        
        # Plot confusion matrix if we have predictions
        if 'predictions' in self.results:
            confusion_plot = f"{config.RESULTS_DIR}/confusion_matrix_{timestamp}.png"
            self._plot_confusion_matrix(confusion_plot)
        
        # Plot training curves if available
        if hasattr(self.model.model, 'evals_result_') and self.model.model.evals_result_:
            training_plot = f"{config.RESULTS_DIR}/training_curves_{timestamp}.png"
            self._plot_training_curves(training_plot)
        else:
            logger.info("No training curves available (model trained without validation set)")
        
        logger.info("Visualizations generated successfully!")
    
    def _plot_confusion_matrix(self, save_path: str):
        """Plot and save confusion matrix"""
        y_true = np.array(self.results['predictions']['y_true'])
        y_pred = np.array(self.results['predictions']['y_pred'])
        
        # Get class names
        class_names = self.preprocessor.target_encoder.classes_
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
    
    def _plot_training_curves(self, save_path: str):
        """Plot training curves"""
        try:
            eval_results = self.model.model.evals_result_
            
            if not eval_results:
                logger.warning("No evaluation results available for plotting")
                return
            
            plt.figure(figsize=(12, 4))
            
            # Plot training and validation loss
            for i, (eval_set, metrics) in enumerate(eval_results.items()):
                plt.subplot(1, 2, i+1)
                for metric_name, values in metrics.items():
                    plt.plot(values, label=f'{eval_set}_{metric_name}')
                plt.title(f'{eval_set} Metrics')
                plt.xlabel('Iteration')
                plt.ylabel('Metric Value')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training curves saved to {save_path}")
            
        except Exception as e:
            logger.warning(f"Could not plot training curves: {str(e)}")
            plt.close()  # Ensure plot is closed even if error occurs
    
    def run_complete_pipeline(self, perform_tuning: bool = True,
                            generate_plots: bool = True) -> dict:
        """
        Run the complete training pipeline
        """
        logger.info("🚀 STARTING COMPLETE TRAINING PIPELINE")
        logger.info("=" * 60)
        
        print("\n" + "="*60)
        print("🔬 SKIN ALLERGY RISK PREDICTION - TRAINING PIPELINE")
        print("="*60)
        
        pipeline_start_time = datetime.now()
        total_steps = 8
        
        try:
            # Step 1: Load data
            print(f"\n📁 Step 1/{total_steps}: Loading Dataset...")
            with tqdm(total=100, desc="Loading Data", ncols=100) as pbar:
                df = self.load_data()
                pbar.update(100)
            print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
            
            # Step 2: Preprocess data
            print(f"\n🔧 Step 2/{total_steps}: Preprocessing Data...")
            with tqdm(total=100, desc="Data Preprocessing", ncols=100) as pbar:
                X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data(df)
                pbar.update(100)
            print(f"✅ Data preprocessed: Train({X_train.shape[0]}), Val({X_val.shape[0]}), Test({X_test.shape[0]})")
            
            # Combine train and val for cross-validation
            X_full_train = pd.concat([X_train, X_val])
            y_full_train = np.concatenate([y_train, y_val])
            
            # Step 3: Train model
            print(f"\n🤖 Step 3/{total_steps}: Training Model...")
            self.train_model(X_train, y_train, X_val, y_val, perform_tuning)
            
            # Step 4: Evaluate model
            print(f"\n📊 Step 4/{total_steps}: Evaluating Model...")
            with tqdm(total=100, desc="Model Evaluation", ncols=100) as pbar:
                self.evaluate_model(X_test, y_test)
                pbar.update(100)
            print("✅ Model evaluation completed")
            
            # Step 5: Cross-validation
            print(f"\n🔄 Step 5/{total_steps}: Cross-Validation...")
            with tqdm(total=100, desc="Cross-Validation", ncols=100) as pbar:
                self.perform_cross_validation(X_full_train, y_full_train)
                pbar.update(100)
            print("✅ Cross-validation completed")
            
            # Step 6: Feature importance analysis
            print(f"\n🎯 Step 6/{total_steps}: Analyzing Feature Importance...")
            with tqdm(total=100, desc="Feature Analysis", ncols=100) as pbar:
                self.analyze_feature_importance()
                pbar.update(100)
            print("✅ Feature importance analysis completed")
            
            # Step 7: Save results
            print(f"\n💾 Step 7/{total_steps}: Saving Results...")
            with tqdm(total=100, desc="Saving Results", ncols=100) as pbar:
                results_file = self.save_results()
                pbar.update(100)
            print(f"✅ Results saved to: {results_file}")
            
            # Step 8: Generate visualizations
            if generate_plots:
                print(f"\n📈 Step 8/{total_steps}: Generating Visualizations...")
                with tqdm(total=100, desc="Creating Plots", ncols=100) as pbar:
                    self.generate_visualizations()
                    pbar.update(100)
                print("✅ Visualizations generated")
            else:
                print(f"\n⏭️  Step 8/{total_steps}: Skipping Visualizations...")
            
            pipeline_end_time = datetime.now()
            total_time = pipeline_end_time - pipeline_start_time
            
            print("\n" + "="*60)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"⏱️  Total execution time: {total_time}")
            
            # Display summary metrics
            if 'test_metrics' in self.results:
                test_metrics = self.results['test_metrics']
                print(f"\n📋 FINAL RESULTS SUMMARY:")
                print(f"   🎯 Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
                print(f"   📊 Test F1 Score: {test_metrics.get('f1_weighted', 0):.4f}")
                print(f"   📈 Test ROC AUC: {test_metrics.get('roc_auc_ovr', 0):.4f}")
            
            if 'cross_validation' in self.results:
                cv_results = self.results['cross_validation']
                print(f"\n🔄 CROSS-VALIDATION RESULTS:")
                for metric, scores in cv_results.items():
                    if isinstance(scores, np.ndarray):
                        print(f"   📊 CV {metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            print(f"\n💾 Results saved to: {results_file}")
            print("="*60)
            
            logger.info("=" * 60)
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Total time: {total_time}")
            logger.info(f"Results saved to: {results_file}")
            logger.info("=" * 60)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            print(f"\n❌ Pipeline failed with error: {str(e)}")
            raise


class TestingPipeline:
    """
    Testing pipeline for evaluating trained models on new data
    """
    
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        """Initialize testing pipeline"""
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = XGBoostSkinAllergyModel()
        self.preprocessor = DataPreprocessor()
        self.evaluator = ModelEvaluator()
        
        logger.info("Testing Pipeline initialized")
    
    def load_trained_model(self):
        """Load pre-trained model and preprocessor"""
        logger.info("Loading trained model and preprocessor...")
        
        # Load model
        if self.model_path:
            self.model.load_model(self.model_path)
        else:
            self.model.load_model()  # Use default path
        
        # Load preprocessor
        if self.preprocessor_path:
            self.preprocessor.load_preprocessor(self.preprocessor_path)
        else:
            self.preprocessor.load_preprocessor()  # Use default path
        
        logger.info("Model and preprocessor loaded successfully!")
    
    def predict_on_new_data(self, new_data: pd.DataFrame) -> dict:
        """Make predictions on new data"""
        logger.info(f"Making predictions on {len(new_data)} samples...")
        
        # Preprocess new data
        X_processed = self.preprocessor.transform(new_data)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)
        
        # Convert predictions back to original labels
        pred_labels = self.preprocessor.target_encoder.inverse_transform(predictions)
        
        results = {
            'predictions': pred_labels.tolist(),
            'probabilities': probabilities.tolist(),
            'risk_levels': pred_labels.tolist(),
            'confidence_scores': np.max(probabilities, axis=1).tolist()
        }
        
        logger.info("Predictions completed successfully!")
        return results
    
    def evaluate_on_test_data(self, test_data: pd.DataFrame) -> dict:
        """Evaluate model on test data with known labels"""
        logger.info("Evaluating model on test data...")
        
        # Prepare data
        X, y = self.preprocessor.prepare_features_target(test_data)
        X_processed = self.preprocessor.transform(test_data)
        y_encoded = self.preprocessor.encode_target(y, fit=False)
        
        # Evaluate
        metrics, y_pred, y_pred_proba = self.model.evaluate(X_processed, y_encoded)
        
        # Generate evaluation report
        evaluation_results = self.evaluator.generate_evaluation_report(
            y_encoded, y_pred, y_pred_proba,
            self.preprocessor.target_encoder.classes_
        )
        
        return evaluation_results


def main():
    """Main function to run the training pipeline"""
    # Initialize training pipeline
    pipeline = TrainingPipeline()
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        perform_tuning=True,
        generate_plots=True
    )
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE SUMMARY")
    print("="*60)
    
    if 'test_metrics' in results:
        test_metrics = results['test_metrics']
        print(f"Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
        print(f"Test F1 Score: {test_metrics.get('f1_weighted', 0):.4f}")
        print(f"Test ROC AUC: {test_metrics.get('roc_auc_ovr', 0):.4f}")
    
    if 'cross_validation' in results:
        cv_results = results['cross_validation']
        for metric, scores in cv_results.items():
            if isinstance(scores, np.ndarray):
                print(f"CV {metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    print("\nModel training completed successfully!")


if __name__ == "__main__":
    main()
