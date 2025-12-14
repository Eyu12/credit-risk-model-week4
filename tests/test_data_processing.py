
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_processing import (
    DateTimeFeatureExtractor,
    CustomerAggregateFeatures,
    DataProcessor
)

class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        """Create test data"""
        # Create sample transaction data
        self.sample_data = pd.DataFrame({
            'TransactionId': [f'TX{i}' for i in range(10)],
            'CustomerId': ['C1', 'C1', 'C1', 'C2', 'C2', 'C3', 'C3', 'C3', 'C3', 'C4'],
            'TransactionStartTime': pd.date_range('2024-01-01', periods=10, freq='h'),
            'Amount': [100, 200, -50, 500, 600, 50, 75, 100, -25, 1000],
            'Value': [100, 200, 50, 500, 600, 50, 75, 100, 25, 1000],
            'ProductCategory': ['airtime', 'financial', 'airtime', 'utility', 'financial', 
                               'airtime', 'airtime', 'data', 'financial', 'ticket'],
            'ProviderId': ['P1', 'P2', 'P1', 'P3', 'P2', 'P1', 'P1', 'P3', 'P2', 'P4'],
            'ChannelId': ['CH1', 'CH2', 'CH1', 'CH3', 'CH2', 'CH1', 'CH1', 'CH3', 'CH2', 'CH4'],
            'FraudResult': [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            'PricingStrategy': [2, 2, 4, 2, 2, 1, 2, 2, 4, 2]
        })
    
    def test_datetime_feature_extractor(self):
        """Test datetime feature extraction"""
        transformer = DateTimeFeatureExtractor()
        transformed = transformer.fit_transform(self.sample_data)
        
        # Check that new features were created
        expected_features = ['transaction_hour', 'transaction_day', 'transaction_month',
                            'transaction_year', 'transaction_dayofweek']
        
        for feature in expected_features:
            self.assertIn(feature, transformed.columns)
        
        # Check specific values
        self.assertEqual(transformed['transaction_hour'].iloc[0], 0)  # First hour
    
    def test_customer_aggregate_features(self):
        """Test customer aggregate feature creation"""
        transformer = CustomerAggregateFeatures()
        transformer.fit(self.sample_data)
        transformed = transformer.transform(self.sample_data)
        
        # Check that customer aggregate features were created
        customer_features = [col for col in transformed.columns if 'customer_' in col]
        self.assertGreater(len(customer_features), 0)
        
        # Check specific customer calculations
        c1_mask = transformed['CustomerId'] == 'C1'
        self.assertEqual(transformed.loc[c1_mask, 'customer_transaction_count'].iloc[0], 3)
        self.assertAlmostEqual(transformed.loc[c1_mask, 'customer_total_amount'].iloc[0], 250)
    
    def test_data_processor_integration(self):
        """Test the complete data processor"""
        processor = DataProcessor()
        processed_data = processor.fit_transform(self.sample_data)
        
        # Check that features were created
        self.assertGreater(processed_data.shape[1], self.sample_data.shape[1])
        
        # Check that feature columns were stored
        self.assertIsNotNone(processor.feature_columns)
        self.assertGreater(len(processor.feature_columns), 0)
    
    def test_negative_amount_patterns(self):
        """Test negative amount (fee) pattern detection"""
        from src.data_processing import NegativeAmountPatterns
        
        transformer = NegativeAmountPatterns()
        transformer.fit(self.sample_data)
        transformed = transformer.transform(self.sample_data)
        
        # Check fee-related features
        self.assertIn('customer_fee_count', transformed.columns)
        self.assertIn('is_current_fee', transformed.columns)
        
        # Customer C1 should have 1 fee transaction
        c1_mask = transformed['CustomerId'] == 'C1'
        self.assertEqual(transformed.loc[c1_mask, 'customer_fee_count'].iloc[0], 1)
    
    def test_feature_importance(self):
        """Test feature importance calculation"""
        processor = DataProcessor()
        processed_data = processor.fit_transform(self.sample_data)
        importance_df = processor.get_feature_importance()
        
        # For small sample, feature importance might be empty
        # Just check that the method runs without error
        self.assertIsInstance(importance_df, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()