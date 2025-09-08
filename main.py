"""
Main entry point for the Skin Allergy Risk Prediction project
Provides command-line interface for training, testing, and running the API
"""

import argparse
import sys
import os
import logging
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from train_test import TrainingPipeline, TestingPipeline

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL),
                   format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

def train_model(args):
    """Train the XGBoost model"""
    print("🚀 TRAINING SKIN ALLERGY PREDICTION MODEL")
    print("=" * 50)
    
    try:
        # Initialize training pipeline
        data_file = args.data_file if args.data_file else config.DATASET_FILE
        pipeline = TrainingPipeline(data_file)
        
        # Run training pipeline
        results = pipeline.run_complete_pipeline(
            perform_tuning=not args.no_tuning,
            generate_plots=not args.no_plots
        )
        
        # Print summary
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        if 'test_metrics' in results:
            test_metrics = results['test_metrics']
            print(f"📊 Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
            print(f"📊 F1 Score: {test_metrics.get('f1_weighted', 0):.4f}")
            print(f"📊 ROC AUC: {test_metrics.get('roc_auc_ovr_weighted', 0):.4f}")
        
        print(f"💾 Model saved to: {config.MODELS_DIR}/")
        print(f"📋 Results saved to: {config.RESULTS_DIR}/")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

def test_model(args):
    """Test the trained model"""
    print("🧪 TESTING SKIN ALLERGY PREDICTION MODEL")
    print("=" * 50)
    
    try:
        # Initialize testing pipeline
        pipeline = TestingPipeline(
            model_path=args.model_path,
            preprocessor_path=args.preprocessor_path
        )
        
        # Load trained model
        pipeline.load_trained_model()
        
        if args.test_data:
            # Test on provided data
            import pandas as pd
            test_data = pd.read_csv(args.test_data)
            
            results = pipeline.evaluate_on_test_data(test_data)
            
            print("📊 Test Results:")
            basic_metrics = results.get('basic_metrics', {})
            print(f"  Accuracy: {basic_metrics.get('accuracy', 0):.4f}")
            print(f"  F1 Score: {basic_metrics.get('f1_weighted', 0):.4f}")
            print(f"  Precision: {basic_metrics.get('precision_weighted', 0):.4f}")
            print(f"  Recall: {basic_metrics.get('recall_weighted', 0):.4f}")
            
        else:
            print("ℹ️  No test data provided. Use --test-data to specify test file.")
            print("✅ Model loaded successfully and ready for predictions!")
        
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        sys.exit(1)

def run_api(args):
    """Run the Flask API server"""
    print("🌐 STARTING SKIN ALLERGY PREDICTION API")
    print("=" * 50)
    
    try:
        # Initialize the API
        if initialize_app():
            host = args.host if args.host else config.API_HOST
            port = args.port if args.port else config.API_PORT
            debug = args.debug if args.debug is not None else config.API_DEBUG
            
            print(f"🚀 Starting Flask API server...")
            print(f"🌐 API URL: http://{host}:{port}")
            print(f"📚 API Info: http://{host}:{port}/")
            print(f"❤️  Health check: http://{host}:{port}/health")
            print(f"🔮 Prediction: POST http://{host}:{port}/predict")
            print(f"💡 Note: Use Streamlit app for interactive interface")
            print("=" * 50)
            
            app.run(host=host, port=port, debug=debug)
        else:
            print("❌ Failed to initialize API")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"API startup failed: {str(e)}")
        sys.exit(1)

def run_streamlit_app(args):
    """Run the Streamlit web application"""
    print("🚀 STARTING STREAMLIT WEB APPLICATION")
    print("=" * 50)
    
    try:
        import subprocess
        import sys
        
        # Build streamlit command
        cmd = [
            sys.executable, 
            "-m", 
            "streamlit", 
            "run", 
            "streamlit_app.py"
        ]
        
        # Add port if specified
        if args.port:
            cmd.extend(["--server.port", str(args.port)])
        
        # Add host if specified
        if args.host:
            cmd.extend(["--server.address", args.host])
        
        print(f"🌐 Starting Streamlit app...")
        print(f"🏠 URL will be: http://{args.host or 'localhost'}:{args.port or 8501}")
        print("=" * 50)
        
        # Run streamlit
        subprocess.run(cmd)
        
    except ImportError:
        print("❌ Streamlit not installed. Install with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Streamlit startup failed: {str(e)}")
        sys.exit(1)

def show_project_info():
    """Show project information"""
    print("🧴 SKIN ALLERGY RISK PREDICTION PROJECT")
    print("=" * 50)
    print(f"Version: {config.VERSION}")
    print(f"Model: XGBoost Classifier")
    print(f"Purpose: Predict skin allergy risk based on personal and environmental factors")
    print("\nAvailable Commands:")
    print("  train           - Train the XGBoost model")
    print("  test            - Test the trained model")
    print("  api             - Start the Flask web API server")
    print("  streamlit       - Start the Streamlit web application")
    print("  info            - Show this information")
    print("\nRecommended Interface:")
    print("  🌟 streamlit    - Modern, interactive web interface (RECOMMENDED)")
    print("  🔧 api          - REST API for programmatic access")
    print("\nProject Structure:")
    print("  📁 data/        - Dataset files")
    print("  📁 models/      - Trained model files")
    print("  📁 results/     - Training results and plots")
    print("  📁 logs/        - Log files")
    print("\nFor help with specific commands, use: python main.py <command> --help")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Skin Allergy Risk Prediction - XGBoost ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data-file', type=str,
                             help='Path to training data CSV file')
    train_parser.add_argument('--no-tuning', action='store_true',
                             help='Skip hyperparameter tuning')
    train_parser.add_argument('--no-plots', action='store_true',
                             help='Skip generating plots')
    
    # Testing command
    test_parser = subparsers.add_parser('test', help='Test the trained model')
    test_parser.add_argument('--test-data', type=str,
                            help='Path to test data CSV file')
    test_parser.add_argument('--model-path', type=str,
                            help='Path to trained model file')
    test_parser.add_argument('--preprocessor-path', type=str,
                            help='Path to preprocessor file')
    
    # API command
    api_parser = subparsers.add_parser('api', help='Start the Flask web API server')
    api_parser.add_argument('--host', type=str, default='127.0.0.1',
                           help='Host to bind the server (default: 127.0.0.1)')
    api_parser.add_argument('--port', type=int, default=5000,
                           help='Port to bind the server (default: 5000)')
    api_parser.add_argument('--debug', action='store_true',
                           help='Run in debug mode')
    
    # Streamlit command
    streamlit_parser = subparsers.add_parser('streamlit', help='Start the Streamlit web application')
    streamlit_parser.add_argument('--host', type=str, default='localhost',
                                 help='Host to bind the server (default: localhost)')
    streamlit_parser.add_argument('--port', type=int, default=8501,
                                 help='Port to bind the server (default: 8501)')
    
    # Info command
    subparsers.add_parser('info', help='Show project information')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'train':
        train_model(args)
    elif args.command == 'test':
        test_model(args)
    elif args.command == 'api':
        run_api(args)
    elif args.command == 'streamlit':
        run_streamlit_app(args)
    elif args.command == 'info':
        show_project_info()
    else:
        # No command provided, show help
        parser.print_help()
        print("\n" + "="*60)
        show_project_info()

if __name__ == "__main__":
    main()
