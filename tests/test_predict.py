
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Now import from src
from src.predict import RFMCalculator, CustomerClusterer, HighRiskLabeler

class TestTargetEngineering(unittest.TestCase):
    
    def setUp(self):
        """Create test data"""
        # Create sample transaction data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        customers = ['C001', 'C002', 'C003', 'C004', 'C005'] * 20
        
        self.sample_data = pd.DataFrame({
            'TransactionId': [f'TX{i:03d}' for i in range(100)],
            'CustomerId': customers,
            'TransactionStartTime': dates,
            'Amount': np.random.randn(100) * 100 + 500,
            'Value': np.random.randn(100) * 100 + 500,
            'ProductCategory': np.random.choice(['airtime', 'financial', 'utility'], 100),
            'ProviderId': np.random.choice(['P1', 'P2', 'P3', 'P4'], 100),
            'ChannelId': np.random.choice(['CH1', 'CH2', 'CH3'], 100)
        })
        
        # Add some negative amounts (fees)
        fee_indices = np.random.choice(100, 20, replace=False)
        self.sample_data.loc[fee_indices, 'Amount'] = -abs(self.sample_data.loc[fee_indices, 'Amount']) * 0.1
    
    def test_rfm_calculator(self):
        """Test RFM calculation"""
        calculator = RFMCalculator()
        rfm_data = calculator.fit_transform(self.sample_data)
        
        # Check that RFM data was created
        self.assertIsNotNone(rfm_data)
        self.assertGreater(len(rfm_data), 0)
        
        # Check required columns
        required_columns = ['recency', 'frequency', 'monetary']
        for col in required_columns:
            self.assertIn(col, rfm_data.columns)
        
        # Check that we have one row per unique customer
        unique_customers = self.sample_data['CustomerId'].nunique()
        self.assertEqual(len(rfm_data), unique_customers)
        
        # Check that recency is non-negative
        self.assertTrue((rfm_data['recency'] >= 0).all())
        
        # Check that frequency is positive
        self.assertTrue((rfm_data['frequency'] > 0).all())
    
    def test_rfm_segments(self):
        """Test RFM segmentation"""
        calculator = RFMCalculator()
        rfm_data = calculator.fit_transform(self.sample_data)
        rfm_segments = calculator.get_rfm_segments(r=3, f=3, m=3)
        
        # Check that segmentation columns were added
        segmentation_columns = ['R_Score', 'F_Score', 'M_Score', 'RFM_Score', 'Segment']
        for col in segmentation_columns:
            self.assertIn(col, rfm_segments.columns)
        
        # Check that all customers have a segment
        self.assertEqual(len(rfm_segments), len(rfm_data))
        self.assertFalse(rfm_segments['Segment'].isna().any())
    
    def test_customer_clusterer(self):
        """Test customer clustering"""
        calculator = RFMCalculator()
        rfm_data = calculator.fit_transform(self.sample_data)
        
        clusterer = CustomerClusterer(n_clusters=3, method='kmeans', random_state=42)
        cluster_labels = clusterer.fit_predict(rfm_data)
        
        # Check that cluster labels were created
        self.assertIsNotNone(cluster_labels)
        self.assertEqual(len(cluster_labels), len(rfm_data))
        
        # Check that we have the expected number of clusters
        unique_clusters = np.unique(cluster_labels)
        self.assertLessEqual(len(unique_clusters), 3)  # Could be less if some clusters are empty
        
        # Check that high-risk cluster was identified
        self.assertTrue(hasattr(clusterer, 'high_risk_cluster'))
        self.assertIsInstance(clusterer.high_risk_cluster, (int, np.integer))
    
    def test_high_risk_labeler_single_cluster(self):
        """Test high-risk labeling with single cluster"""
        calculator = RFMCalculator()
        rfm_data = calculator.fit_transform(self.sample_data)
        
        clusterer = CustomerClusterer(n_clusters=3, method='kmeans', random_state=42)
        cluster_labels = clusterer.fit_predict(rfm_data)
        
        labeler = HighRiskLabeler(high_risk_cluster=clusterer.high_risk_cluster)
        high_risk_labels = labeler.create_labels(rfm_data, cluster_labels)
        
        # Check that labels were created
        self.assertIsNotNone(high_risk_labels)
        self.assertEqual(len(high_risk_labels), len(rfm_data))
        
        # Check that labels are binary
        unique_labels = np.unique(high_risk_labels)
        self.assertTrue(set(unique_labels).issubset({0, 1}))
        
        # Check that some customers are labeled as high-risk
        self.assertGreater(high_risk_labels.sum(), 0)
        
        # Check that not all customers are high-risk
        self.assertLess(high_risk_labels.sum(), len(high_risk_labels))
    
    def test_high_risk_labeler_multiple_criteria(self):
        """Test high-risk labeling with multiple criteria"""
        calculator = RFMCalculator()
        rfm_data = calculator.fit_transform(self.sample_data)
        
        clusterer = CustomerClusterer(n_clusters=3, method='kmeans', random_state=42)
        cluster_labels = clusterer.fit_predict(rfm_data)
        
        labeler = HighRiskLabeler(high_risk_cluster=clusterer.high_risk_cluster, 
                                 risk_threshold=0.5)
        high_risk_labels = labeler.create_labels(rfm_data, cluster_labels, 
                                                use_multiple_criteria=True)
        
        # Check that labels were created
        self.assertIsNotNone(high_risk_labels)
        self.assertEqual(len(high_risk_labels), len(rfm_data))
        
        # Check that labels are binary
        unique_labels = np.unique(high_risk_labels)
        self.assertTrue(set(unique_labels).issubset({0, 1}))
        
        # Check that risk scores were calculated
        self.assertTrue(hasattr(labeler, 'risk_scores'))
        self.assertEqual(len(labeler.risk_scores), len(rfm_data))
    
    def test_high_risk_analysis(self):
        """Test high-risk customer analysis"""
        calculator = RFMCalculator()
        rfm_data = calculator.fit_transform(self.sample_data)
        
        clusterer = CustomerClusterer(n_clusters=3, method='kmeans', random_state=42)
        cluster_labels = clusterer.fit_predict(rfm_data)
        
        labeler = HighRiskLabeler(high_risk_cluster=clusterer.high_risk_cluster)
        high_risk_labels = labeler.create_labels(rfm_data, cluster_labels)
        
        # Analyze high-risk customers
        analysis_results = labeler.analyze_high_risk_customers(rfm_data, high_risk_labels)
        
        # Check that analysis was performed
        self.assertIsNotNone(analysis_results)
        self.assertIsInstance(analysis_results, pd.DataFrame)
        
        # Check that we have statistics for both risk groups
        self.assertEqual(len(analysis_results), 2)  # 0 and 1
    
    def test_integration(self):
        """Test the complete integration"""
        from src.predict import TargetVariableEngineer
        
        target_engineer = TargetVariableEngineer(
            n_clusters=3,
            clustering_method='kmeans',
            use_multiple_criteria=False,
            random_state=42
        )
        
        rfm_with_labels, high_risk_labels = target_engineer.fit_transform(self.sample_data)
        
        # Check outputs
        self.assertIsNotNone(rfm_with_labels)
        self.assertIsNotNone(high_risk_labels)
        
        # Check that columns were added
        self.assertIn('cluster', rfm_with_labels.columns)
        self.assertIn('is_high_risk', rfm_with_labels.columns)
        
        # Check that labels match
        self.assertTrue((rfm_with_labels['is_high_risk'] == high_risk_labels.values).all())
        
        # Test merging back to original data
        data_with_risk = target_engineer.merge_with_original_data(
            self.sample_data, 
            high_risk_labels
        )
        
        self.assertIn('is_high_risk', data_with_risk.columns)
        self.assertEqual(len(data_with_risk), len(self.sample_data))

if __name__ == '__main__':
    unittest.main()