import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Any
import logging
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# For WoE transformation
from category_encoders import WOEEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ensure_directory_exists(file_path: str):
    """Ensure the directory for a file exists"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract datetime features from transaction timestamp"""
    
    def __init__(self, datetime_col: str = 'TransactionStartTime'):
        self.datetime_col = datetime_col
        self.feature_names = []
        
    def fit(self, X: pd.DataFrame, y=None):
        # No fitting required for this transformer
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Extracting datetime features...")
        
        # Ensure datetime format
        X_copy = X.copy()
        X_copy[self.datetime_col] = pd.to_datetime(X_copy[self.datetime_col])
        
        # Extract features
        X_copy['transaction_hour'] = X_copy[self.datetime_col].dt.hour
        X_copy['transaction_day'] = X_copy[self.datetime_col].dt.day
        X_copy['transaction_month'] = X_copy[self.datetime_col].dt.month
        X_copy['transaction_year'] = X_copy[self.datetime_col].dt.year
        X_copy['transaction_dayofweek'] = X_copy[self.datetime_col].dt.dayofweek
        X_copy['transaction_weekofyear'] = X_copy[self.datetime_col].dt.isocalendar().week
        X_copy['transaction_quarter'] = X_copy[self.datetime_col].dt.quarter
        
        # Cyclical encoding for hour (captures 24-hour cycle)
        X_copy['hour_sin'] = np.sin(2 * np.pi * X_copy['transaction_hour']/24)
        X_copy['hour_cos'] = np.cos(2 * np.pi * X_copy['transaction_hour']/24)
        
        # Is business hours feature (9 AM to 5 PM)
        X_copy['is_business_hours'] = ((X_copy['transaction_hour'] >= 9) & 
                                       (X_copy['transaction_hour'] <= 17)).astype(int)
        
        self.feature_names = ['transaction_hour', 'transaction_day', 'transaction_month',
                             'transaction_year', 'transaction_dayofweek', 'transaction_weekofyear',
                             'transaction_quarter', 'hour_sin', 'hour_cos', 'is_business_hours']
        
        logger.info(f"Extracted {len(self.feature_names)} datetime features")
        return X_copy


class CustomerAggregateFeatures(BaseEstimator, TransformerMixin):
    """Create aggregate features at customer level"""
    
    def __init__(self, customer_col: str = 'CustomerId', amount_col: str = 'Amount'):
        self.customer_col = customer_col
        self.amount_col = amount_col
        self.customer_stats = {}
        self.feature_names = []
        
    def fit(self, X: pd.DataFrame, y=None):
        logger.info("Fitting customer aggregate features...")
        
        # Calculate customer-level statistics
        customer_groups = X.groupby(self.customer_col)
        
        self.customer_stats = {
            'total_amount': customer_groups[self.amount_col].sum(),
            'avg_amount': customer_groups[self.amount_col].mean(),
            'transaction_count': customer_groups[self.amount_col].count(),
            'std_amount': customer_groups[self.amount_col].std(),
            'min_amount': customer_groups[self.amount_col].min(),
            'max_amount': customer_groups[self.amount_col].max(),
            'median_amount': customer_groups[self.amount_col].median(),
            'amount_range': customer_groups[self.amount_col].max() - customer_groups[self.amount_col].min()
        }
        
        # Fill NaN values for std (where only one transaction)
        self.customer_stats['std_amount'] = self.customer_stats['std_amount'].fillna(0)
        
        self.feature_names = ['total_amount', 'avg_amount', 'transaction_count',
                             'std_amount', 'min_amount', 'max_amount',
                             'median_amount', 'amount_range']
        
        logger.info(f"Calculated customer stats for {len(self.customer_stats['total_amount'])} unique customers")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Transforming with customer aggregate features...")
        
        X_copy = X.copy()
        
        # Map customer-level statistics to each transaction
        for feature_name, stats_series in self.customer_stats.items():
            X_copy[f'customer_{feature_name}'] = X_copy[self.customer_col].map(stats_series)
        
        # Create additional derived features
        X_copy['customer_amount_ratio_to_avg'] = X_copy[self.amount_col] / X_copy['customer_avg_amount']
        X_copy['customer_amount_zscore'] = (X_copy[self.amount_col] - X_copy['customer_avg_amount']) / X_copy['customer_std_amount'].replace(0, 1)
        
        # Flag for outlier transactions (beyond 2 standard deviations)
        X_copy['customer_amount_outlier'] = (abs(X_copy['customer_amount_zscore']) > 2).astype(int)
        
        # Transaction velocity (time between transactions would require full dataset)
        # For now, we'll create a simple transaction frequency feature
        if 'transaction_year' in X_copy.columns:
            year_range = X_copy['transaction_year'].max() - X_copy['transaction_year'].min() + 1
            if year_range > 0:
                X_copy['customer_transaction_frequency'] = X_copy['customer_transaction_count'] / year_range
            else:
                X_copy['customer_transaction_frequency'] = X_copy['customer_transaction_count']
        
        logger.info(f"Added {len([col for col in X_copy.columns if 'customer_' in col])} customer aggregate features")
        return X_copy


class ProviderProductFeatures(BaseEstimator, TransformerMixin):
    """Create features based on provider and product combinations"""
    
    def __init__(self, provider_col: str = 'ProviderId', 
                 product_col: str = 'ProductCategory',
                 fraud_col: str = 'FraudResult'):
        self.provider_col = provider_col
        self.product_col = product_col
        self.fraud_col = fraud_col
        self.provider_risk = {}
        self.product_risk = {}
        self.feature_names = []
        
    def fit(self, X: pd.DataFrame, y=None):
        logger.info("Fitting provider and product risk features...")
        
        # Calculate provider fraud rate
        if self.fraud_col in X.columns:
            provider_groups = X.groupby(self.provider_col)[self.fraud_col]
            self.provider_risk = {
                'fraud_rate': provider_groups.mean(),
                'fraud_count': provider_groups.sum(),
                'total_transactions': provider_groups.count()
            }
        
        # Calculate product category fraud rate
        if self.fraud_col in X.columns and self.product_col in X.columns:
            product_groups = X.groupby(self.product_col)[self.fraud_col]
            self.product_risk = {
                'fraud_rate': product_groups.mean(),
                'fraud_count': product_groups.sum(),
                'total_transactions': product_groups.count()
            }
        
        logger.info(f"Calculated risk for {len(self.provider_risk.get('fraud_rate', []))} providers "
                   f"and {len(self.product_risk.get('fraud_rate', []))} product categories")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Transforming with provider/product features...")
        
        X_copy = X.copy()
        
        # Map provider risk features
        if self.provider_risk and self.provider_col in X.columns:
            X_copy['provider_fraud_rate'] = X_copy[self.provider_col].map(self.provider_risk['fraud_rate']).fillna(0)
            X_copy['provider_fraud_count'] = X_copy[self.provider_col].map(self.provider_risk['fraud_count']).fillna(0)
            X_copy['provider_total_transactions'] = X_copy[self.provider_col].map(self.provider_risk['total_transactions']).fillna(0)
        
        # Map product risk features
        if self.product_risk and self.product_col in X.columns:
            X_copy['product_fraud_rate'] = X_copy[self.product_col].map(self.product_risk['fraud_rate']).fillna(0)
            X_copy['product_fraud_count'] = X_copy[self.product_col].map(self.product_risk['fraud_count']).fillna(0)
            X_copy['product_total_transactions'] = X_copy[self.product_col].map(self.product_risk['total_transactions']).fillna(0)
        
        # Interaction features
        if 'provider_fraud_rate' in X_copy.columns and 'product_fraud_rate' in X_copy.columns:
            X_copy['provider_product_risk_score'] = X_copy['provider_fraud_rate'] * X_copy['product_fraud_rate']
        
        logger.info(f"Added {len([col for col in X_copy.columns if 'provider_' in col or 'product_' in col])} provider/product features")
        return X_copy


class ChannelFeatures(BaseEstimator, TransformerMixin):
    """Create features based on transaction channel"""
    
    def __init__(self, channel_col: str = 'ChannelId', fraud_col: str = 'FraudResult'):
        self.channel_col = channel_col
        self.fraud_col = fraud_col
        self.channel_stats = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        logger.info("Fitting channel features...")
        
        if self.fraud_col in X.columns and self.channel_col in X.columns:
            channel_groups = X.groupby(self.channel_col)[self.fraud_col]
            self.channel_stats = {
                'fraud_rate': channel_groups.mean(),
                'fraud_count': channel_groups.sum(),
                'total_transactions': channel_groups.count(),
                'avg_amount': X.groupby(self.channel_col)['Amount'].mean() if 'Amount' in X.columns else pd.Series()
            }
        
        logger.info(f"Calculated stats for {len(self.channel_stats.get('fraud_rate', []))} channels")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Transforming with channel features...")
        
        X_copy = X.copy()
        
        if self.channel_stats and self.channel_col in X.columns:
            X_copy['channel_fraud_rate'] = X_copy[self.channel_col].map(self.channel_stats['fraud_rate']).fillna(0)
            X_copy['channel_total_transactions'] = X_copy[self.channel_col].map(self.channel_stats['total_transactions']).fillna(0)
            
            if 'avg_amount' in self.channel_stats and not self.channel_stats['avg_amount'].empty:
                X_copy['channel_avg_amount'] = X_copy[self.channel_col].map(self.channel_stats['avg_amount']).fillna(0)
                if 'Amount' in X.columns:
                    X_copy['amount_vs_channel_avg'] = X_copy['Amount'] / X_copy['channel_avg_amount'].replace(0, 1)
        
        logger.info(f"Added {len([col for col in X_copy.columns if 'channel_' in col])} channel features")
        return X_copy


class FinancialPatternFeatures(BaseEstimator, TransformerMixin):
    """Create financial pattern features"""
    
    def __init__(self, amount_col: str = 'Amount', value_col: str = 'Value'):
        self.amount_col = amount_col
        self.value_col = value_col
        
    def fit(self, X: pd.DataFrame, y=None):
        # No fitting required
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating financial pattern features...")
        
        X_copy = X.copy()
        
        # Basic financial features
        if self.amount_col in X.columns and self.value_col in X.columns:
            X_copy['amount_value_ratio'] = X_copy[self.amount_col] / X_copy[self.value_col].replace(0, 1)
            X_copy['amount_value_diff'] = X_copy[self.amount_col] - X_copy[self.value_col]
            
            # Flag for fee transactions (negative amounts)
            X_copy['is_fee_transaction'] = (X_copy[self.amount_col] < 0).astype(int)
            X_copy['fee_amount'] = X_copy[self.amount_col].apply(lambda x: abs(x) if x < 0 else 0)
            
            # Transaction size categories
            X_copy['transaction_size_category'] = pd.cut(
                abs(X_copy[self.amount_col]),
                bins=[0, 100, 500, 1000, 5000, 10000, float('inf')],
                labels=['micro', 'small', 'medium', 'large', 'xlarge', 'xxlarge']
            )
        
        # Amount-based features
        if self.amount_col in X.columns:
            X_copy['amount_abs'] = X_copy[self.amount_col].abs()
            X_copy['amount_log'] = np.log1p(X_copy['amount_abs'])
            X_copy['amount_sqrt'] = np.sqrt(X_copy['amount_abs'])
            
            # Decile of transaction amount
            try:
                X_copy['amount_decile'] = pd.qcut(X_copy['amount_abs'], q=10, labels=False, duplicates='drop')
            except:
                # If qcut fails, use equal-width binning
                X_copy['amount_decile'] = pd.cut(X_copy['amount_abs'], bins=10, labels=False)
        
        logger.info(f"Added financial pattern features")
        return X_copy


class NegativeAmountPatterns(BaseEstimator, TransformerMixin):
    """Analyze patterns in negative amounts (fees/charges)"""
    
    def __init__(self, customer_col: str = 'CustomerId', amount_col: str = 'Amount'):
        self.customer_col = customer_col
        self.amount_col = amount_col
        self.customer_fee_stats = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        logger.info("Fitting negative amount patterns...")
        
        # Identify negative transactions (fees/charges)
        negative_mask = X[self.amount_col] < 0
        negative_transactions = X[negative_mask]
        
        if not negative_transactions.empty:
            customer_groups = negative_transactions.groupby(self.customer_col)
            
            self.customer_fee_stats = {
                'total_fees': customer_groups[self.amount_col].sum().abs(),
                'avg_fee': customer_groups[self.amount_col].mean().abs(),
                'fee_count': customer_groups[self.amount_col].count(),
                'fee_frequency': customer_groups[self.amount_col].count() / X.groupby(self.customer_col)[self.amount_col].count()
            }
        
        logger.info(f"Analyzed fee patterns for {len(self.customer_fee_stats.get('total_fees', []))} customers")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Transforming with negative amount patterns...")
        
        X_copy = X.copy()
        
        # Map customer fee statistics
        if self.customer_fee_stats:
            X_copy['customer_total_fees'] = X_copy[self.customer_col].map(self.customer_fee_stats['total_fees']).fillna(0)
            X_copy['customer_avg_fee'] = X_copy[self.customer_col].map(self.customer_fee_stats['avg_fee']).fillna(0)
            X_copy['customer_fee_count'] = X_copy[self.customer_col].map(self.customer_fee_stats['fee_count']).fillna(0)
            X_copy['customer_fee_frequency'] = X_copy[self.customer_col].map(self.customer_fee_stats['fee_frequency']).fillna(0)
            
            # Fee to transaction ratio
            if 'customer_total_amount' in X_copy.columns:
                X_copy['fee_to_transaction_ratio'] = X_copy['customer_total_fees'] / X_copy['customer_total_amount'].abs().replace(0, 1)
        
        # Current transaction fee features
        X_copy['is_current_fee'] = (X_copy[self.amount_col] < 0).astype(int)
        X_copy['current_fee_amount'] = X_copy[self.amount_col].apply(lambda x: abs(x) if x < 0 else 0)
        
        logger.info(f"Added {len([col for col in X_copy.columns if 'fee' in col])} fee-related features")
        return X_copy


class WoETransformer(BaseEstimator, TransformerMixin):
    """Weight of Evidence transformation for categorical variables"""
    
    def __init__(self, categorical_cols: List[str] = None, target_col: str = 'FraudResult'):
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.woe_encoder = None
        self.feature_names = []
        self.iv_scores = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        logger.info("Fitting WoE transformer...")
        
        if self.categorical_cols is None:
            # Automatically identify categorical columns
            self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if self.target_col in X.columns and self.categorical_cols:
            try:
                # Initialize WOE encoder
                self.woe_encoder = WOEEncoder(cols=self.categorical_cols)
                
                # Fit WOE encoder
                self.woe_encoder.fit(X[self.categorical_cols], X[self.target_col])
                
                self.feature_names = [f'{col}_woe' for col in self.categorical_cols]
                
                # Get IV scores if available
                if hasattr(self.woe_encoder, 'feature_importances_'):
                    for col, iv in zip(self.categorical_cols, self.woe_encoder.feature_importances_):
                        self.iv_scores[col] = {
                            'iv': iv,
                            'category': self._categorize_iv(iv)
                        }
                
                logger.info(f"Fitted WoE for {len(self.categorical_cols)} categorical columns")
            except Exception as e:
                logger.warning(f"Error fitting WoE transformer: {str(e)}")
                self.woe_encoder = None
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Transforming with WoE...")
        
        X_copy = X.copy()
        
        if self.woe_encoder is not None and self.categorical_cols:
            try:
                # Apply WOE transformation
                woe_transformed = self.woe_encoder.transform(X_copy[self.categorical_cols])
                
                # Rename columns
                woe_transformed.columns = [f'{col}_woe' for col in self.categorical_cols]
                
                # Add WoE features to dataframe
                X_copy = pd.concat([X_copy, woe_transformed], axis=1)
                
                # Log IV scores
                if self.iv_scores:
                    logger.info("Information Value (IV) Scores:")
                    for col, iv_info in self.iv_scores.items():
                        logger.info(f"  {col}: {iv_info['iv']:.4f} ({iv_info['category']})")
            except Exception as e:
                logger.warning(f"Error applying WoE transformation: {str(e)}")
        
        return X_copy
    
    def _categorize_iv(self, iv: float) -> str:
        """Categorize Information Value based on predictive power"""
        if iv < 0.02:
            return "Not useful for prediction"
        elif iv < 0.1:
            return "Weak predictive power"
        elif iv < 0.3:
            return "Medium predictive power"
        elif iv < 0.5:
            return "Strong predictive power"
        else:
            return "Suspicious predictive power (check for data leakage)"


class DataProcessor:
    """Main data processing pipeline"""
    
    def __init__(self, target_col: str = 'FraudResult', data_dir: str = None):
        self.target_col = target_col
        self.pipeline = None
        self.feature_columns = None
        self.data_dir = data_dir or self._get_default_data_dir()
        self._build_pipeline()
    
    def _get_default_data_dir(self):
        """Get default data directory path"""
        # Assuming the script is in src directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        return os.path.join(project_root, 'data')
    
    def _build_pipeline(self):
        """Build the complete feature engineering pipeline"""
        
        # Define column groups
        self.categorical_cols = ['ProductCategory', 'ChannelId', 'ProviderId']
        self.numerical_cols = ['Amount', 'Value', 'PricingStrategy']
        
        logger.info("Building feature engineering pipeline...")
        
        # Create pipeline steps
        pipeline_steps = [
            ('datetime_features', DateTimeFeatureExtractor()),
            ('customer_features', CustomerAggregateFeatures()),
            ('provider_product_features', ProviderProductFeatures()),
            ('channel_features', ChannelFeatures()),
            ('financial_patterns', FinancialPatternFeatures()),
            ('negative_amount_patterns', NegativeAmountPatterns()),
            ('woe_transformation', WoETransformer(categorical_cols=['ProductCategory', 'ProviderId']))
        ]
        
        self.pipeline = Pipeline(pipeline_steps)
        logger.info("Pipeline built successfully")
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data"""
        logger.info("Starting data processing pipeline...")
        
        # Make a copy to avoid modifying original data
        data_copy = data.copy()
        
        # Fit and transform
        processed_data = self.pipeline.fit_transform(data_copy)
        
        # Store feature columns (excluding target and ID columns)
        id_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
        self.feature_columns = [col for col in processed_data.columns 
                               if col not in id_cols + [self.target_col] + ['TransactionStartTime']]
        
        logger.info(f"Data processing complete. Created {len(self.feature_columns)} features")
        logger.info(f"Final dataset shape: {processed_data.shape}")
        
        return processed_data
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline"""
        if self.pipeline is None:
            raise ValueError("Pipeline must be fitted first. Call fit_transform()")
        
        logger.info("Transforming new data...")
        return self.pipeline.transform(data.copy())
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance based on IV scores (for categorical) and variance (for numerical)"""
        feature_importance = []
        
        # Check if WoE transformer has IV scores
        if hasattr(self.pipeline, 'named_steps'):
            woe_step = self.pipeline.named_steps.get('woe_transformation')
            if woe_step and hasattr(woe_step, 'iv_scores') and woe_step.iv_scores:
                for col, iv_info in woe_step.iv_scores.items():
                    feature_importance.append({
                        'feature': f'{col}_woe',
                        'importance': iv_info['iv'],
                        'type': 'categorical',
                        'importance_type': 'IV',
                        'category': iv_info['category']
                    })
        
        # Create DataFrame and sort only if we have data
        if feature_importance:
            df = pd.DataFrame(feature_importance)
            return df.sort_values('importance', ascending=False)
        else:
            logger.info("No feature importance data available")
            return pd.DataFrame(columns=['feature', 'importance', 'type', 'importance_type', 'category'])
    
    def get_feature_summary(self) -> Dict[str, int]:
        """Get summary of feature types created"""
        if not self.feature_columns:
            return {}
        
        feature_types = {}
        for col in self.feature_columns:
            if 'woe' in col:
                feature_types['WoE Features'] = feature_types.get('WoE Features', 0) + 1
            elif 'customer_' in col:
                feature_types['Customer Features'] = feature_types.get('Customer Features', 0) + 1
            elif 'provider_' in col or 'product_' in col:
                feature_types['Provider/Product Features'] = feature_types.get('Provider/Product Features', 0) + 1
            elif 'channel_' in col:
                feature_types['Channel Features'] = feature_types.get('Channel Features', 0) + 1
            elif 'fee' in col:
                feature_types['Fee Features'] = feature_types.get('Fee Features', 0) + 1
            elif 'transaction_' in col and 'customer_' not in col:
                feature_types['Temporal Features'] = feature_types.get('Temporal Features', 0) + 1
            else:
                feature_types['Other Features'] = feature_types.get('Other Features', 0) + 1
        
        return feature_types
    
    def save_processed_data(self, data: pd.DataFrame, output_path: str = None):
        """Save processed data to file"""
        if output_path is None:
            output_path = os.path.join(self.data_dir, 'processed', 'features_engineered.csv')
        
        ensure_directory_exists(output_path)
        logger.info(f"Saving processed data to {output_path}")
        data.to_csv(output_path, index=False)
        return output_path
    
    def load_processed_data(self, input_path: str = None) -> pd.DataFrame:
        """Load processed data from file"""
        if input_path is None:
            input_path = os.path.join(self.data_dir, 'processed', 'features_engineered.csv')
        
        logger.info(f"Loading processed data from {input_path}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Processed data file not found: {input_path}")
        
        return pd.read_csv(input_path)


def create_sample_data():
    """Create sample data for testing if no real data exists"""
    logger.info("Creating sample data for testing...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample data with similar structure
    sample_data = pd.DataFrame({
        'TransactionId': range(1, n_samples + 1),
        'BatchId': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'AccountId': np.random.choice([1001, 1002, 1003, 1004, 1005], n_samples),
        'SubscriptionId': np.random.choice([2001, 2002, 2003, 2004, 2005], n_samples),
        'CustomerId': np.random.choice(['C001', 'C002', 'C003', 'C004', 'C005', 'C006', 'C007', 'C008', 'C009', 'C010'], n_samples),
        'TransactionStartTime': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'Amount': np.random.uniform(-100, 1000, n_samples),
        'Value': np.random.uniform(0, 1200, n_samples),
        'ProductCategory': np.random.choice(['Electronics', 'Clothing', 'Food', 'Services', 'Entertainment'], n_samples),
        'ChannelId': np.random.choice(['Web', 'Mobile', 'POS', 'ATM'], n_samples),
        'ProviderId': np.random.choice(['P001', 'P002', 'P003', 'P004'], n_samples),
        'PricingStrategy': np.random.choice([1, 2, 3], n_samples),
        'FraudResult': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    })
    
    return sample_data


def main():
    """Main function to run data processing"""
    
    # Create the data directory structure
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')
    
    # Ensure directories exist
    for directory in [data_dir, raw_dir, processed_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    
    # Example usage
    try:
        # Define paths
        raw_data_path = os.path.join(raw_dir, 'data.csv')
        
        # Check if raw data exists
        if os.path.exists(raw_data_path):
            logger.info(f"Loading raw data from {raw_data_path}")
            raw_data = pd.read_csv(raw_data_path)
            logger.info(f"Raw data shape: {raw_data.shape}")
        else:
            logger.warning(f"Raw data file not found: {raw_data_path}")
            logger.info("Creating sample data for demonstration...")
            
            # Create sample data
            raw_data = create_sample_data()
            
            # Save sample data
            raw_data.to_csv(raw_data_path, index=False)
            logger.info(f"Saved sample data to {raw_data_path}")
            logger.info(f"Sample data shape: {raw_data.shape}")
        
        # Initialize data processor
        processor = DataProcessor(target_col='FraudResult', data_dir=data_dir)
        
        # Process data
        processed_data = processor.fit_transform(raw_data)
        
        # Save processed data
        processed_data_path = processor.save_processed_data(processed_data)
        logger.info(f"Processed data saved to {processed_data_path}")
        
        # Get feature importance
        feature_importance = processor.get_feature_importance()
        if not feature_importance.empty:
            logger.info("\nTop Features by Information Value:")
            print(feature_importance.to_string())
        else:
            logger.info("\nNo feature importance data available (IV scores not calculated)")
        
        # Summary of created features
        logger.info(f"\nTotal features created: {len(processor.feature_columns)}")
        
        feature_types = processor.get_feature_summary()
        if feature_types:
            logger.info("Feature types:")
            for feature_type, count in feature_types.items():
                logger.info(f"  {feature_type}: {count}")
        
        # Display first few rows of processed data
        logger.info(f"\nFirst few rows of processed data:")
        print(processed_data.head().to_string())
        
        # Display feature columns
        logger.info(f"\nFirst 20 feature columns:")
        for i, col in enumerate(processor.feature_columns[:20]):
            logger.info(f"  {i+1}. {col}")
        
        if len(processor.feature_columns) > 20:
            logger.info(f"  ... and {len(processor.feature_columns) - 20} more features")
        
        # Save feature list to file
        feature_list_path = os.path.join(processed_dir, 'feature_list.txt')
        with open(feature_list_path, 'w') as f:
            f.write("Feature List:\n")
            f.write("=" * 50 + "\n")
            for i, col in enumerate(processor.feature_columns, 1):
                f.write(f"{i:3d}. {col}\n")
        
        logger.info(f"\nFeature list saved to: {feature_list_path}")
        
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        raise


if __name__ == "__main__":
    main()