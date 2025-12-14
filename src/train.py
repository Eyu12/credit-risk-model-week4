import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import logging
from typing import Tuple, Dict, Any, Optional
import pickle
import json
import os
from pathlib import Path

# Machine learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Create necessary directories before configuring logging
Path("../logs").mkdir(parents=True, exist_ok=True)
Path("../reports").mkdir(parents=True, exist_ok=True)
Path("../artifacts").mkdir(parents=True, exist_ok=True)
Path("../mlruns").mkdir(parents=True, exist_ok=True)
Path("../data/processed").mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def convert_to_serializable(obj):
    """
    Convert numpy/pandas types to JSON serializable Python native types
    
    Parameters:
    -----------
    obj : any
        Object to convert
        
    Returns:
    --------
    JSON serializable object
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj


class ModelTrainer:
    """
    Model training and evaluation with MLflow tracking
    """
    
    def __init__(self, experiment_name: str = "credit-risk-modeling",
                 tracking_uri: str = "../mlruns",
                 random_state: int = 42):
        """
        Initialize model trainer
        
        Parameters:
        -----------
        experiment_name : str
            Name of MLflow experiment
        tracking_uri : str
            URI for MLflow tracking
        random_state : int
            Random seed for reproducibility
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.random_state = random_state
        
        # Initialize MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        # Model registry
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None
        
        logger.info(f"Initialized ModelTrainer with experiment: {experiment_name}")
        logger.info(f"MLflow tracking URI: {tracking_uri}")
    
    def load_and_prepare_data(self, data_path: str, 
                            target_col: str = 'is_high_risk',
                            test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                            pd.Series, pd.Series]:
        """
        Load and prepare data for training
        
        Parameters:
        -----------
        data_path : str
            Path to processed data with target variable
        target_col : str
            Name of target variable column
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Loading data from {data_path}")
        
        # Load data
        data = pd.read_csv(data_path)
        logger.info(f"Data shape: {data.shape}")
        
        # Separate features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Drop non-feature columns
        non_feature_cols = ['TransactionId', 'BatchId', 'AccountId', 
                          'SubscriptionId', 'CustomerId', 'TransactionStartTime']
        for col in non_feature_cols:
            if col in X.columns:
                X = X.drop(columns=[col])
        
        # Check for missing values
        missing_values = X.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Found {missing_values} missing values. Will handle during preprocessing.")
        
        # Check class balance
        class_distribution = y.value_counts(normalize=True)
        logger.info(f"Class distribution: {class_distribution.to_dict()}")
        
        # Split data
        logger.info(f"Splitting data with test_size={test_size}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def create_preprocessing_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create preprocessing pipeline
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training data to infer column types
            
        Returns:
        --------
        ColumnTransformer with preprocessing steps
        """
        logger.info("Creating preprocessing pipeline")
        
        # Identify column types
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Numerical columns: {len(numerical_cols)}")
        logger.info(f"Categorical columns: {len(categorical_cols)}")
        
        # Create transformers
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # For OneHotEncoder, use sparse_output instead of sparse
        try:
            # Try with sparse_output (newer scikit-learn versions)
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
        except TypeError:
            # Fall back to sparse parameter (older scikit-learn versions)
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        return preprocessor
    
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get model configurations for training
        
        Returns:
        --------
        Dictionary of model configurations
        """
        model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'model__C': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'model__penalty': ['l1', 'l2'],
                    'model__solver': ['liblinear', 'saga']
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=self.random_state),
                'params': {
                    'model__max_depth': [3, 5, 7, 10, None],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                'params': {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [5, 10, None],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'model__n_estimators': [50, 100, 200],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__max_depth': [3, 5, 7]
                }
            },
            'xgboost': {
                'model': XGBClassifier(random_state=self.random_state, n_jobs=-1, eval_metric='logloss'),
                'params': {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [3, 5, 7],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__subsample': [0.8, 0.9, 1.0]
                }
            },
            'lightgbm': {
                'model': LGBMClassifier(random_state=self.random_state, n_jobs=-1),
                'params': {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [3, 5, 7, -1],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__num_leaves': [31, 50, 100]
                }
            }
        }
        
        return model_configs
    
    def train_model(self, model_name: str, model_config: Dict[str, Any],
                   X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   preprocessor: ColumnTransformer,
                   use_grid_search: bool = True) -> Dict[str, Any]:
        """
        Train a single model with hyperparameter tuning
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        model_config : dict
            Model configuration
        X_train, y_train : pd.DataFrame, pd.Series
            Training data
        X_test, y_test : pd.DataFrame, pd.Series
            Testing data
        preprocessor : ColumnTransformer
            Preprocessing pipeline
        use_grid_search : bool
            Whether to use GridSearchCV (True) or RandomizedSearchCV (False)
            
        Returns:
        --------
        Dictionary with training results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'='*60}")
        
        with mlflow.start_run(run_name=model_name, nested=True):
            # Create pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model_config['model'])
            ])
            
            # Log parameters
            mlflow.log_param("model", model_name)
            mlflow.log_param("random_state", self.random_state)
            
            # Hyperparameter tuning
            if use_grid_search:
                search = GridSearchCV(
                    pipeline,
                    model_config['params'],
                    cv=5,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                search_method = "GridSearchCV"
            else:
                search = RandomizedSearchCV(
                    pipeline,
                    model_config['params'],
                    n_iter=10,
                    cv=5,
                    scoring='roc_auc',
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbose=1
                )
                search_method = "RandomizedSearchCV"
            
            logger.info(f"Starting {search_method} for {model_name}...")
            
            # Fit the model
            search.fit(X_train, y_train)
            
            # Get best model
            best_model = search.best_estimator_
            best_params = search.best_params_
            
            # Log best parameters
            for param, value in best_params.items():
                mlflow.log_param(param, value)
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Log metrics - convert to float for MLflow
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float, np.number)):
                    mlflow.log_metric(metric_name, float(metric_value))
            
            # Calculate feature importance if available
            feature_importance = self.get_feature_importance(best_model, X_train)
            if feature_importance is not None:
                importance_df = pd.DataFrame(feature_importance, 
                                           columns=['feature', 'importance'])
                importance_path = f"../reports/{model_name}_feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
            
            # Log model
            mlflow.sklearn.log_model(best_model, "model")
            
            # Log confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            cm_path = f"../reports/{model_name}_confusion_matrix.csv"
            pd.DataFrame(cm).to_csv(cm_path)
            mlflow.log_artifact(cm_path)
            
            # Log classification report - ensure JSON serializable
            report = classification_report(y_test, y_pred, output_dict=True)
            report = convert_to_serializable(report)
            report_path = f"../reports/{model_name}_classification_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(report_path)
            
            # Save results
            results = {
                'model_name': model_name,
                'model': best_model,
                'best_params': best_params,
                'best_score': float(search.best_score_) if search.best_score_ is not None else None,
                'metrics': metrics,
                'search_cv_results': search.cv_results_,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best CV score (ROC-AUC): {search.best_score_:.4f}")
            logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"Test F1 Score: {metrics['f1_score']:.4f}")
            
            return results
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        
        Parameters:
        -----------
        y_true : pd.Series
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_pred_proba : np.ndarray
            Predicted probabilities
            
        Returns:
        --------
        Dictionary of metrics with Python native types
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
            'average_precision': float(average_precision_score(y_true, y_pred_proba))
        }
        
        # Add class-specific metrics for binary classification
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics.update({
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp),
                'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            })
        except:
            pass
        
        return metrics
    
    def get_feature_importance(self, model: Pipeline, X: pd.DataFrame) -> Optional[list]:
        """
        Extract feature importance from model
        
        Parameters:
        -----------
        model : Pipeline
            Trained model pipeline
        X : pd.DataFrame
            Training data
            
        Returns:
        --------
        List of (feature, importance) tuples or None
        """
        try:
            # Get the actual model from pipeline
            if hasattr(model, 'named_steps') and 'model' in model.named_steps:
                model_obj = model.named_steps['model']
            else:
                model_obj = model
            
            # Check if model has feature importance
            if hasattr(model_obj, 'feature_importances_'):
                # Get feature names after preprocessing
                preprocessor = model.named_steps['preprocessor']
                
                # Transform X to get feature names
                X_transformed = preprocessor.transform(X)
                
                # Get feature names from column transformer
                feature_names = []
                for name, transformer, columns in preprocessor.transformers_:
                    if name != 'remainder':
                        if hasattr(transformer, 'get_feature_names_out'):
                            feature_names.extend(transformer.get_feature_names_out(columns))
                        else:
                            feature_names.extend(columns)
                
                # Get importances
                importances = model_obj.feature_importances_
                
                # Combine feature names with importances
                importance_list = list(zip(feature_names, importances))
                importance_list.sort(key=lambda x: x[1], reverse=True)
                
                return importance_list[:20]  # Return top 20 features
            
            # Check for coefficients (Logistic Regression)
            elif hasattr(model_obj, 'coef_'):
                preprocessor = model.named_steps['preprocessor']
                X_transformed = preprocessor.transform(X)
                
                feature_names = []
                for name, transformer, columns in preprocessor.transformers_:
                    if name != 'remainder':
                        if hasattr(transformer, 'get_feature_names_out'):
                            feature_names.extend(transformer.get_feature_names_out(columns))
                        else:
                            feature_names.extend(columns)
                
                importances = np.abs(model_obj.coef_[0])
                importance_list = list(zip(feature_names, importances))
                importance_list.sort(key=lambda x: x[1], reverse=True)
                
                return importance_list[:20]
                
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
        
        return None
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        model_names: Optional[list] = None,
                        use_grid_search: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Train multiple models and track experiments
        
        Parameters:
        -----------
        X_train, y_train : pd.DataFrame, pd.Series
            Training data
        X_test, y_test : pd.DataFrame, pd.Series
            Testing data
        model_names : list, optional
            List of model names to train
        use_grid_search : bool
            Whether to use grid search or random search
            
        Returns:
        --------
        Dictionary of training results for all models
        """
        logger.info(f"\n{'='*60}")
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info(f"{'='*60}")
        
        # Create preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline(X_train)
        
        # Get model configurations
        all_model_configs = self.get_model_configs()
        
        # Filter models if specified
        if model_names is None:
            model_names = ['logistic_regression', 'random_forest', 
                          'xgboost', 'lightgbm']
        
        model_configs = {name: all_model_configs[name] for name in model_names 
                        if name in all_model_configs}
        
        logger.info(f"Training models: {list(model_configs.keys())}")
        
        # Train each model
        all_results = {}
        
        with mlflow.start_run(run_name="model_comparison"):
            for model_name, model_config in model_configs.items():
                try:
                    results = self.train_model(
                        model_name=model_name,
                        model_config=model_config,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        preprocessor=preprocessor,
                        use_grid_search=use_grid_search
                    )
                    
                    all_results[model_name] = results
                    
                    # Update best model
                    if results['metrics']['roc_auc'] > self.best_score:
                        self.best_score = results['metrics']['roc_auc']
                        self.best_model = results['model']
                        self.best_model_name = model_name
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    continue
        
        logger.info(f"\nBest model: {self.best_model_name} with ROC-AUC: {self.best_score:.4f}")
        
        return all_results
    
    def register_best_model(self, model_name: str, 
                          X_sample: pd.DataFrame,
                          y_sample: pd.Series):
        """
        Register the best model in MLflow Model Registry
        
        Parameters:
        -----------
        model_name : str
            Name of the model to register
        X_sample : pd.DataFrame
            Sample features for signature inference
        y_sample : pd.Series
            Sample target for signature inference
        """
        if self.best_model is None:
            logger.error("No best model to register")
            return
        
        logger.info(f"Registering best model: {model_name}")
        
        # Infer model signature
        signature = infer_signature(X_sample, self.best_model.predict(X_sample))
        
        # Register model
        mlflow.sklearn.log_model(
            sk_model=self.best_model,
            artifact_path="model",
            signature=signature,
            registered_model_name=f"credit_risk_{model_name}"
        )
        
        # Transition model to Production stage
        client = mlflow.tracking.MlflowClient()
        
        # Get latest version
        model_versions = client.search_model_versions(f"name='credit_risk_{model_name}'")
        if model_versions:
            latest_version = max([mv.version for mv in model_versions])
            client.transition_model_version_stage(
                name=f"credit_risk_{model_name}",
                version=latest_version,
                stage="Production"
            )
            logger.info(f"Model {model_name} v{latest_version} transitioned to Production")
    
    def create_comparison_report(self, all_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Create comparison report of all models
        
        Parameters:
        -----------
        all_results : dict
            Results from training all models
            
        Returns:
        --------
        DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name, results in all_results.items():
            metrics = results['metrics']
            row = {
                'model': model_name,
                'roc_auc': float(metrics['roc_auc']),
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score']),
                'best_cv_score': float(results['best_score']) if results['best_score'] is not None else None
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
        
        # Save comparison report
        report_path = "../reports/model_comparison.csv"
        comparison_df.to_csv(report_path, index=False)
        
        logger.info(f"\nModel Comparison:")
        logger.info(f"\n{comparison_df.to_string()}")
        
        return comparison_df
    
    def save_training_artifacts(self, all_results: Dict[str, Dict[str, Any]]):
        """
        Save training artifacts
        
        Parameters:
        -----------
        all_results : dict
            Results from training all models
        """
        artifacts_dir = Path("../artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        # Save best model
        if self.best_model is not None:
            model_path = artifacts_dir / "best_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            logger.info(f"Saved best model to {model_path}")
        
        # Save all results
        results_path = artifacts_dir / "training_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(all_results, f)
        
        # Save metrics summary with type conversion for JSON serialization
        metrics_summary = {}
        for model_name, results in all_results.items():
            # Convert metrics to JSON serializable format
            metrics_summary[model_name] = convert_to_serializable(results['metrics'])
        
        metrics_path = artifacts_dir / "metrics_summary.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        logger.info(f"Saved training artifacts to {artifacts_dir}")


def main():
    """Main training pipeline"""
    # Initialize trainer
    trainer = ModelTrainer(
        experiment_name="credit-risk-modeling-v1",
        tracking_uri="../mlruns",
        random_state=42
    )
    
    # Load and prepare data
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'data_with_risk_labels.csv')
    logger.info(f"Loading processed data from {data_path}")
    
    # Check if file exists
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at: {data_path}")
        logger.info("Please make sure you have processed data at this location.")
        logger.info("You can create sample data by running a preprocessing script first.")
        return
    
    try:
        X_train, X_test, y_train, y_test = trainer.load_and_prepare_data(
            data_path=data_path,
            target_col='is_high_risk',
            test_size=0.2
        )
        
        # Train models
        all_results = trainer.train_all_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_names=['logistic_regression', 'random_forest', 'xgboost', 'lightgbm'],
            use_grid_search=True  # Use GridSearchCV for thorough search
        )
        
        # Create comparison report
        comparison_df = trainer.create_comparison_report(all_results)
        
        # Register best model
        trainer.register_best_model(
            model_name=trainer.best_model_name,
            X_sample=X_train.head(100),
            y_sample=y_train.head(100)
        )
        
        # Save artifacts
        trainer.save_training_artifacts(all_results)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TRAINING PIPELINE COMPLETED")
        logger.info("="*60)
        logger.info(f"Best Model: {trainer.best_model_name}")
        logger.info(f"Best ROC-AUC: {trainer.best_score:.4f}")
        logger.info(f"Models trained: {len(all_results)}")
        logger.info(f"MLflow UI: mlflow ui --backend-store-uri ../mlruns")
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.info(f"Expected data at: {data_path}")
        logger.info("Please run the preprocessing pipeline first to create the data file.")
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise


if __name__ == "__main__":
    main()