
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
import logging
import os
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import LocalOutlierFactor

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RFMCalculator:
    """Calculate Recency, Frequency, and Monetary metrics for customers"""
    
    def __init__(self, snapshot_date: Optional[str] = None, 
                 customer_col: str = 'CustomerId',
                 datetime_col: str = 'TransactionStartTime',
                 amount_col: str = 'Amount'):
        """
        Initialize RFM calculator
        
        Parameters:
        -----------
        snapshot_date : str, optional
            Reference date for recency calculation (format: 'YYYY-MM-DD')
            If None, uses max date in data
        customer_col : str
            Column name for customer identifier
        datetime_col : str
            Column name for transaction datetime
        amount_col : str
            Column name for transaction amount
        """
        self.snapshot_date = snapshot_date
        self.customer_col = customer_col
        self.datetime_col = datetime_col
        self.amount_col = amount_col
        self.rfm_data = None
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM metrics for all customers
        
        Parameters:
        -----------
        data : pd.DataFrame
            Transaction data with datetime and amount columns
            
        Returns:
        --------
        pd.DataFrame with RFM metrics for each customer
        """
        logger.info("Calculating RFM metrics...")
        
        # Make a copy to avoid modifying original data
        data_copy = data.copy()
        
        # Convert datetime column if needed
        if not pd.api.types.is_datetime64_any_dtype(data_copy[self.datetime_col]):
            data_copy[self.datetime_col] = pd.to_datetime(data_copy[self.datetime_col])
        
        # Determine snapshot date
        if self.snapshot_date is None:
            self.snapshot_date = data_copy[self.datetime_col].max()
        else:
            self.snapshot_date = pd.to_datetime(self.snapshot_date)
        
        logger.info(f"Using snapshot date: {self.snapshot_date}")
        
        # Calculate Recency (days since last transaction)
        recency = data_copy.groupby(self.customer_col)[self.datetime_col].max()
        recency = (self.snapshot_date - recency).dt.days
        
        # Calculate Frequency (number of transactions)
        frequency = data_copy.groupby(self.customer_col).size()
        
        # Calculate Monetary (total transaction amount - use absolute value for analysis)
        monetary = data_copy.groupby(self.customer_col)[self.amount_col].sum().abs()
        
        # Calculate additional metrics
        # Average transaction value
        avg_transaction = data_copy.groupby(self.customer_col)[self.amount_col].mean().abs()
        
        # Transaction consistency (std of days between transactions)
        def calculate_consistency(group):
            if len(group) > 1:
                days_between = group.diff().dt.days.dropna()
                return days_between.std() if not days_between.empty and days_between.std() > 0 else 0
            return 0
        
        consistency = data_copy.groupby(self.customer_col)[self.datetime_col].apply(calculate_consistency)
        
        # Create RFM DataFrame
        self.rfm_data = pd.DataFrame({
            'recency': recency,
            'frequency': frequency,
            'monetary': monetary,
            'avg_transaction': avg_transaction,
            'consistency': consistency
        })
        
        # Calculate additional derived metrics
        self.rfm_data['frequency_recency_ratio'] = self.rfm_data['frequency'] / (self.rfm_data['recency'] + 1)
        self.rfm_data['monetary_per_frequency'] = self.rfm_data['monetary'] / (self.rfm_data['frequency'] + 1)
        self.rfm_data['activity_score'] = np.log1p(self.rfm_data['frequency']) * np.log1p(self.rfm_data['monetary'])
        
        logger.info(f"RFM metrics calculated for {len(self.rfm_data)} unique customers")
        logger.info(f"RFM statistics:\n{self.rfm_data.describe()}")
        
        return self.rfm_data
    
    def get_rfm_segments(self, r: int = 4, f: int = 4, m: int = 4) -> pd.DataFrame:
        """
        Create RFM segments by quantile-based scoring
        
        Parameters:
        -----------
        r, f, m : int
            Number of quantiles for Recency, Frequency, Monetary scoring
            
        Returns:
        --------
        pd.DataFrame with RFM scores and segments
        """
        if self.rfm_data is None:
            raise ValueError("RFM data not calculated. Call fit_transform first.")
        
        logger.info(f"Creating RFM segments with R={r}, F={f}, M={m} quantiles")
        
        # Create RFM scores (lower recency is better, so we reverse it)
        rfm_scored = self.rfm_data.copy()
        
        # Recency: lower is better (more recent), so we use qcut with ascending=True
        # then reverse the scores so 1=worst, r=best
        rfm_scored['R_Score'] = pd.qcut(rfm_scored['recency'], q=r, labels=False, duplicates='drop') + 1
        rfm_scored['R_Score'] = r + 1 - rfm_scored['R_Score']  # Reverse so higher = more recent
        
        # Frequency: higher is better
        rfm_scored['F_Score'] = pd.qcut(rfm_scored['frequency'], q=f, labels=False, duplicates='drop') + 1
        
        # Monetary: higher is better
        rfm_scored['M_Score'] = pd.qcut(rfm_scored['monetary'], q=m, labels=False, duplicates='drop') + 1
        
        # Calculate RFM score
        rfm_scored['RFM_Score'] = rfm_scored['R_Score'] + rfm_scored['F_Score'] + rfm_scored['M_Score']
        
        # Create RFM cell (string concatenation)
        rfm_scored['RFM_Cell'] = rfm_scored['R_Score'].astype(str) + rfm_scored['F_Score'].astype(str) + rfm_scored['M_Score'].astype(str)
        
        # Define segments based on RFM scores
        def assign_segment(row):
            if row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
                return 'Champions'
            elif row['R_Score'] >= 3 and row['F_Score'] >= 2:
                return 'Loyal Customers'
            elif row['R_Score'] >= 3:
                return 'Recent Customers'
            elif row['R_Score'] <= 2 and row['F_Score'] <= 2 and row['M_Score'] <= 2:
                return 'At Risk'
            elif row['R_Score'] <= 2 and row['F_Score'] <= 2:
                return 'Hibernating'
            else:
                return 'Need Attention'
        
        rfm_scored['Segment'] = rfm_scored.apply(assign_segment, axis=1)
        
        logger.info(f"Segment distribution:\n{rfm_scored['Segment'].value_counts()}")
        
        return rfm_scored


class CustomerClusterer:
    """Cluster customers based on RFM metrics to identify risk segments"""
    
    def __init__(self, n_clusters: int = 3, 
                 method: str = 'kmeans',
                 random_state: int = 42,
                 use_pca: bool = False,
                 pca_components: Optional[int] = None):
        """
        Initialize customer clusterer
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create
        method : str
            Clustering method: 'kmeans', 'gmm', 'agglomerative', 'dbscan'
        random_state : int
            Random seed for reproducibility
        use_pca : bool
            Whether to use PCA for dimensionality reduction
        pca_components : int, optional
            Number of PCA components to use
        """
        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.scaler = StandardScaler()
        self.pca = None
        self.clusterer = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.cluster_stats = None
        
    def fit_predict(self, rfm_data: pd.DataFrame) -> pd.Series:
        """
        Fit clustering model and predict clusters
        
        Parameters:
        -----------
        rfm_data : pd.DataFrame
            RFM metrics for customers
            
        Returns:
        --------
        pd.Series with cluster labels for each customer
        """
        logger.info(f"Clustering customers using {self.method} with {self.n_clusters} clusters")
        
        # Select features for clustering
        clustering_features = ['recency', 'frequency', 'monetary']
        
        # Add additional features if available
        additional_features = ['avg_transaction', 'consistency', 'activity_score']
        for feature in additional_features:
            if feature in rfm_data.columns:
                clustering_features.append(feature)
        
        X = rfm_data[clustering_features].copy()
        
        # Handle any missing values
        X = X.fillna(X.mean())
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA if requested
        if self.use_pca:
            if self.pca_components is None:
                self.pca_components = min(X.shape[1], 3)
            
            self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            X_scaled = self.pca.fit_transform(X_scaled)
            logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_}")
        
        # Choose clustering method
        if self.method == 'kmeans':
            self.clusterer = KMeans(n_clusters=self.n_clusters, 
                                   random_state=self.random_state,
                                   n_init=10)
        elif self.method == 'gmm':
            self.clusterer = GaussianMixture(n_components=self.n_clusters,
                                            random_state=self.random_state)
        elif self.method == 'agglomerative':
            self.clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
        elif self.method == 'dbscan':
            self.clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unsupported clustering method: {self.method}")
        
        # Fit and predict
        if self.method == 'gmm':
            self.cluster_labels = self.clusterer.fit_predict(X_scaled)
            self.cluster_centers = self.clusterer.means_
        elif self.method == 'dbscan':
            self.cluster_labels = self.clusterer.fit_predict(X_scaled)
            self.n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        else:
            self.cluster_labels = self.clusterer.fit_predict(X_scaled)
            if hasattr(self.clusterer, 'cluster_centers_'):
                self.cluster_centers = self.clusterer.cluster_centers_
        
        # Calculate clustering metrics
        if len(set(self.cluster_labels)) > 1 and -1 not in self.cluster_labels:
            try:
                silhouette = silhouette_score(X_scaled, self.cluster_labels)
                calinski = calinski_harabasz_score(X_scaled, self.cluster_labels)
                davies = davies_bouldin_score(X_scaled, self.cluster_labels)
                
                logger.info(f"Clustering metrics:")
                logger.info(f"  Silhouette Score: {silhouette:.3f} (higher is better)")
                logger.info(f"  Calinski-Harabasz Score: {calinski:.3f} (higher is better)")
                logger.info(f"  Davies-Bouldin Score: {davies:.3f} (lower is better)")
            except:
                logger.warning("Could not calculate clustering metrics")
        
        # Create cluster labels series
        cluster_series = pd.Series(self.cluster_labels, index=rfm_data.index, name='cluster')
        
        # Analyze cluster characteristics
        self._analyze_clusters(rfm_data, cluster_series)
        
        return cluster_series
    
    def _analyze_clusters(self, rfm_data: pd.DataFrame, cluster_labels: pd.Series):
        """Analyze and describe each cluster"""
        logger.info("Analyzing cluster characteristics...")
        
        # Add cluster labels to data
        data_with_clusters = rfm_data.copy()
        data_with_clusters['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = data_with_clusters.groupby('cluster').agg({
            'recency': ['mean', 'std', 'count'],
            'frequency': ['mean', 'std'],
            'monetary': ['mean', 'std'],
            'avg_transaction': ['mean', 'std'] if 'avg_transaction' in data_with_clusters.columns else None
        }).round(2)
        
        self.cluster_stats = cluster_stats
        
        logger.info("Cluster Statistics:")
        print(cluster_stats)
        
        # Identify risk clusters
        self._identify_risk_clusters(data_with_clusters)
    
    def _identify_risk_clusters(self, data_with_clusters: pd.DataFrame):
        """Identify high-risk clusters based on RFM patterns"""
        
        # High risk typically characterized by:
        # - High recency (inactive for long time)
        # - Low frequency
        # - Low monetary value
        
        cluster_means = data_with_clusters.groupby('cluster')[['recency', 'frequency', 'monetary']].mean()
        
        # Normalize metrics for scoring
        cluster_normalized = cluster_means.copy()
        for col in ['recency', 'frequency', 'monetary']:
            cluster_normalized[col] = (cluster_means[col] - cluster_means[col].min()) / \
                                     (cluster_means[col].max() - cluster_means[col].min() + 1e-10)
        
        # Risk score: high recency + low frequency + low monetary
        # We invert frequency and monetary since lower is worse
        cluster_normalized['risk_score'] = (
            cluster_normalized['recency'] +  # Higher recency = worse
            (1 - cluster_normalized['frequency']) +  # Lower frequency = worse
            (1 - cluster_normalized['monetary'])  # Lower monetary = worse
        )
        
        # Sort by risk score (higher = more risky)
        cluster_risk_ranking = cluster_normalized['risk_score'].sort_values(ascending=False)
        
        logger.info("\nCluster Risk Ranking (higher score = higher risk):")
        for cluster, score in cluster_risk_ranking.items():
            logger.info(f"  Cluster {cluster}: {score:.3f}")
        
        self.risk_ranking = cluster_risk_ranking
        
        # Identify high-risk cluster (highest risk score)
        self.high_risk_cluster = cluster_risk_ranking.index[0]
        
        logger.info(f"\nIdentified high-risk cluster: {self.high_risk_cluster}")
        logger.info(f"Cluster characteristics:")
        for col in ['recency', 'frequency', 'monetary']:
            logger.info(f"  {col}: {cluster_means.loc[self.high_risk_cluster, col]:.2f}")
    
    def visualize_clusters(self, rfm_data: pd.DataFrame, cluster_labels: pd.Series, 
                          save_path: Optional[str] = None):
        """
        Visualize the clusters
        
        Parameters:
        -----------
        rfm_data : pd.DataFrame
            RFM metrics
        cluster_labels : pd.Series
            Cluster assignments
        save_path : str, optional
            Path to save the visualization
        """
        logger.info("Creating cluster visualizations...")
        
        # Prepare data for visualization
        viz_data = rfm_data.copy()
        viz_data['cluster'] = cluster_labels.values
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 3D Scatter plot (if we have 3+ features)
        if len(viz_data.columns) >= 3:
            ax1 = fig.add_subplot(231, projection='3d')
            
            # Use first three numerical columns
            numeric_cols = viz_data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'cluster']
            
            if len(numeric_cols) >= 3:
                ax1.scatter(viz_data[numeric_cols[0]], 
                          viz_data[numeric_cols[1]], 
                          viz_data[numeric_cols[2]],
                          c=viz_data['cluster'], 
                          cmap='viridis', 
                          alpha=0.6)
                ax1.set_xlabel(numeric_cols[0])
                ax1.set_ylabel(numeric_cols[1])
                ax1.set_zlabel(numeric_cols[2])
                ax1.set_title('3D Cluster Visualization')
        
        # 2. Recency vs Frequency scatter
        ax2 = fig.add_subplot(232)
        scatter2 = ax2.scatter(viz_data['recency'], viz_data['frequency'], 
                              c=viz_data['cluster'], cmap='viridis', alpha=0.6)
        ax2.set_xlabel('Recency (days)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Recency vs Frequency by Cluster')
        plt.colorbar(scatter2, ax=ax2)
        
        # 3. Frequency vs Monetary scatter
        ax3 = fig.add_subplot(233)
        scatter3 = ax3.scatter(viz_data['frequency'], viz_data['monetary'], 
                              c=viz_data['cluster'], cmap='viridis', alpha=0.6)
        ax3.set_xlabel('Frequency')
        ax3.set_ylabel('Monetary')
        ax3.set_title('Frequency vs Monetary by Cluster')
        plt.colorbar(scatter3, ax=ax3)
        
        # 4. Box plots for each cluster
        ax4 = fig.add_subplot(234)
        box_data = []
        cluster_ids = sorted(viz_data['cluster'].unique())
        for cluster_id in cluster_ids:
            box_data.append(viz_data[viz_data['cluster'] == cluster_id]['recency'].values)
        ax4.boxplot(box_data, labels=[f'Cluster {c}' for c in cluster_ids])
        ax4.set_ylabel('Recency (days)')
        ax4.set_title('Recency Distribution by Cluster')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Cluster size bar chart
        ax5 = fig.add_subplot(235)
        cluster_sizes = viz_data['cluster'].value_counts().sort_index()
        bars = ax5.bar(range(len(cluster_sizes)), cluster_sizes.values)
        ax5.set_xlabel('Cluster')
        ax5.set_ylabel('Number of Customers')
        ax5.set_title('Cluster Sizes')
        ax5.set_xticks(range(len(cluster_sizes)))
        ax5.set_xticklabels([f'Cluster {c}' for c in cluster_sizes.index])
        
        # Color high-risk cluster red
        if hasattr(self, 'high_risk_cluster'):
            for i, (cluster_id, bar) in enumerate(zip(cluster_sizes.index, bars)):
                if cluster_id == self.high_risk_cluster:
                    bar.set_color('red')
                    bar.set_label('High Risk')
        
        # 6. RFM radar chart for cluster centroids
        if hasattr(self, 'cluster_centers') and self.cluster_centers is not None:
            ax6 = fig.add_subplot(236, polar=True)
            
            # Use first 4 features for radar chart
            n_features = min(4, self.cluster_centers.shape[1])
            features = ['recency', 'frequency', 'monetary', 'avg_transaction'][:n_features]
            
            # Normalize for radar chart
            centers_normalized = self.cluster_centers.copy()
            for i in range(n_features):
                centers_normalized[:, i] = (self.cluster_centers[:, i] - self.cluster_centers[:, i].min()) / \
                                         (self.cluster_centers[:, i].max() - self.cluster_centers[:, i].min() + 1e-10)
            
            angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
            angles += angles[:1]  # Close the polygon
            
            for cluster_id in range(self.n_clusters):
                values = centers_normalized[cluster_id, :n_features].tolist()
                values += values[:1]
                ax6.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster_id}')
                ax6.fill(angles, values, alpha=0.25)
            
            ax6.set_xticks(angles[:-1])
            ax6.set_xticklabels(features)
            ax6.set_title('Cluster Centroids (Normalized)')
            ax6.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()


class HighRiskLabeler:
    """Create high-risk labels based on clustering results"""
    
    def __init__(self, high_risk_cluster: Optional[int] = None,
                 risk_threshold: float = 0.3):
        """
        Initialize high-risk labeler
        
        Parameters:
        -----------
        high_risk_cluster : int, optional
            Cluster ID to label as high-risk
            If None, will be determined automatically
        risk_threshold : float
            Percentage threshold for labeling high-risk (if using multiple criteria)
        """
        self.high_risk_cluster = high_risk_cluster
        self.risk_threshold = risk_threshold
        self.label_mapping = {}
        
    def create_labels(self, rfm_data: pd.DataFrame, 
                     cluster_labels: pd.Series,
                     use_multiple_criteria: bool = False) -> pd.Series:
        """
        Create binary high-risk labels
        
        Parameters:
        -----------
        rfm_data : pd.DataFrame
            RFM metrics
        cluster_labels : pd.Series
            Cluster assignments
        use_multiple_criteria : bool
            Whether to use additional criteria beyond clustering
            
        Returns:
        --------
        pd.Series with binary labels (1 = high risk, 0 = not high risk)
        """
        logger.info("Creating high-risk labels...")
        
        if use_multiple_criteria:
            labels = self._create_labels_multiple_criteria(rfm_data, cluster_labels)
        else:
            labels = self._create_labels_single_cluster(cluster_labels)
        
        # Analyze label distribution
        label_counts = labels.value_counts()
        label_percentage = label_counts / len(labels) * 100
        
        logger.info("\nHigh-Risk Label Distribution:")
        logger.info(f"  Not High-Risk (0): {label_counts.get(0, 0):,} customers ({label_percentage.get(0, 0):.2f}%)")
        logger.info(f"  High-Risk (1): {label_counts.get(1, 0):,} customers ({label_percentage.get(1, 0):.2f}%)")
        
        return labels
    
    def _create_labels_single_cluster(self, cluster_labels: pd.Series) -> pd.Series:
        """Create labels based on single cluster"""
        if self.high_risk_cluster is None:
            raise ValueError("high_risk_cluster must be specified for single cluster labeling")
        
        labels = (cluster_labels == self.high_risk_cluster).astype(int)
        labels.name = 'is_high_risk'
        
        self.label_mapping = {
            'method': 'single_cluster',
            'high_risk_cluster': self.high_risk_cluster,
            'label_distribution': labels.value_counts().to_dict()
        }
        
        return labels
    
    def _create_labels_multiple_criteria(self, rfm_data: pd.DataFrame, 
                                        cluster_labels: pd.Series) -> pd.Series:
        """Create labels using multiple risk criteria"""
        logger.info("Using multiple criteria for high-risk labeling")
        
        # Combine RFM data with cluster labels
        data_with_clusters = rfm_data.copy()
        data_with_clusters['cluster'] = cluster_labels.values
        
        # Calculate risk score for each customer
        risk_scores = pd.Series(0.0, index=rfm_data.index)
        
        # 1. Recency risk (higher recency = higher risk)
        recency_normalized = (data_with_clusters['recency'] - data_with_clusters['recency'].min()) / \
                           (data_with_clusters['recency'].max() - data_with_clusters['recency'].min() + 1e-10)
        risk_scores += recency_normalized
        
        # 2. Frequency risk (lower frequency = higher risk)
        frequency_normalized = 1 - ((data_with_clusters['frequency'] - data_with_clusters['frequency'].min()) / \
                                  (data_with_clusters['frequency'].max() - data_with_clusters['frequency'].min() + 1e-10))
        risk_scores += frequency_normalized
        
        # 3. Monetary risk (lower monetary = higher risk)
        monetary_normalized = 1 - ((data_with_clusters['monetary'] - data_with_clusters['monetary'].min()) / \
                                 (data_with_clusters['monetary'].max() - data_with_clusters['monetary'].min() + 1e-10))
        risk_scores += monetary_normalized
        
        # 4. Cluster risk (if high_risk_cluster is specified)
        if self.high_risk_cluster is not None:
            cluster_risk = (data_with_clusters['cluster'] == self.high_risk_cluster).astype(float) * 3
            risk_scores += cluster_risk
        
        # Normalize risk scores
        risk_scores_normalized = (risk_scores - risk_scores.min()) / \
                               (risk_scores.max() - risk_scores.min() + 1e-10)
        
        # Label high-risk customers based on threshold
        labels = (risk_scores_normalized >= self.risk_threshold).astype(int)
        labels.name = 'is_high_risk'
        
        self.label_mapping = {
            'method': 'multiple_criteria',
            'risk_threshold': self.risk_threshold,
            'average_risk_score': risk_scores_normalized.mean(),
            'label_distribution': labels.value_counts().to_dict()
        }
        
        # Save risk scores for analysis
        self.risk_scores = risk_scores_normalized
        
        return labels
    
    def analyze_high_risk_customers(self, rfm_data: pd.DataFrame, 
                                   labels: pd.Series) -> pd.DataFrame:
        """
        Analyze characteristics of high-risk customers
        
        Parameters:
        -----------
        rfm_data : pd.DataFrame
            RFM metrics
        labels : pd.Series
            High-risk labels
            
        Returns:
        --------
        DataFrame with comparison of high-risk vs not-high-risk groups
        """
        logger.info("\nAnalyzing high-risk customer characteristics...")
        
        data_with_labels = rfm_data.copy()
        data_with_labels['is_high_risk'] = labels.values
        
        # Group statistics
        group_stats = data_with_labels.groupby('is_high_risk').agg({
            'recency': ['mean', 'std', 'min', 'max'],
            'frequency': ['mean', 'std', 'min', 'max'],
            'monetary': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        logger.info("High-Risk vs Not-High-Risk Comparison:")
        print(group_stats)
        
        return group_stats


class TargetVariableEngineer:
    """Main class for engineering proxy target variable"""
    
    def __init__(self, n_clusters: int = 3,
                 clustering_method: str = 'kmeans',
                 use_multiple_criteria: bool = False,
                 risk_threshold: float = 0.3,
                 random_state: int = 42):
        """
        Initialize target variable engineer
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters for segmentation
        clustering_method : str
            Clustering algorithm to use
        use_multiple_criteria : bool
            Whether to use multiple criteria for labeling
        risk_threshold : float
            Threshold for high-risk labeling
        random_state : int
            Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.use_multiple_criteria = use_multiple_criteria
        self.risk_threshold = risk_threshold
        self.random_state = random_state
        
        self.rfm_calculator = None
        self.clusterer = None
        self.labeler = None
        self.high_risk_labels = None
        
    def fit_transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create proxy target variable for credit risk
        
        Parameters:
        -----------
        data : pd.DataFrame
            Transaction data
            
        Returns:
        --------
        Tuple of (RFM data with clusters, high-risk labels)
        """
        logger.info("="*60)
        logger.info("PROXY TARGET VARIABLE ENGINEERING")
        logger.info("="*60)
        
        # Step 1: Calculate RFM metrics
        logger.info("\nStep 1: Calculating RFM metrics...")
        self.rfm_calculator = RFMCalculator()
        rfm_data = self.rfm_calculator.fit_transform(data)
        
        # Step 2: Cluster customers
        logger.info("\nStep 2: Clustering customers...")
        self.clusterer = CustomerClusterer(
            n_clusters=self.n_clusters,
            method=self.clustering_method,
            random_state=self.random_state
        )
        cluster_labels = self.clusterer.fit_predict(rfm_data)
        
        # Step 3: Create high-risk labels
        logger.info("\nStep 3: Creating high-risk labels...")
        self.labeler = HighRiskLabeler(
            high_risk_cluster=getattr(self.clusterer, 'high_risk_cluster', None),
            risk_threshold=self.risk_threshold
        )
        high_risk_labels = self.labeler.create_labels(
            rfm_data, 
            cluster_labels,
            use_multiple_criteria=self.use_multiple_criteria
        )
        
        # Step 4: Add labels to RFM data
        rfm_data_with_labels = rfm_data.copy()
        rfm_data_with_labels['cluster'] = cluster_labels.values
        rfm_data_with_labels['is_high_risk'] = high_risk_labels.values
        
        self.high_risk_labels = high_risk_labels
        
        # Step 5: Analyze results
        logger.info("\nStep 4: Analyzing high-risk customers...")
        self.labeler.analyze_high_risk_customers(rfm_data, high_risk_labels)
        
        logger.info("\n" + "="*60)
        logger.info("TARGET VARIABLE ENGINEERING COMPLETE")
        logger.info("="*60)
        
        return rfm_data_with_labels, high_risk_labels
    
    def visualize_results(self, rfm_data: pd.DataFrame, save_dir: str = '../reports'):
        """Create visualizations of the results"""
        import os
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Visualize clusters
        if self.clusterer and hasattr(self.clusterer, 'cluster_labels'):
            self.clusterer.visualize_clusters(
                rfm_data,
                pd.Series(self.clusterer.cluster_labels, index=rfm_data.index),
                save_path=os.path.join(save_dir, 'cluster_visualization.png')
            )
        
        # Create additional visualizations
        self._create_risk_visualizations(rfm_data, save_dir)
    
    def _create_risk_visualizations(self, rfm_data: pd.DataFrame, save_dir: str):
        """Create risk-specific visualizations"""
        if self.high_risk_labels is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Risk distribution
        risk_counts = self.high_risk_labels.value_counts()
        colors = ['lightgreen', 'lightcoral']
        axes[0, 0].pie(risk_counts.values, labels=['Not High-Risk', 'High-Risk'], 
                      colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('High-Risk Label Distribution')
        
        # 2. RFM comparison by risk
        data_with_labels = rfm_data.copy()
        data_with_labels['is_high_risk'] = self.high_risk_labels.values
        
        # Recency comparison
        risk_groups = data_with_labels.groupby('is_high_risk')['recency']
        axes[0, 1].boxplot([risk_groups.get_group(0), risk_groups.get_group(1)], 
                          labels=['Not High-Risk', 'High-Risk'])
        axes[0, 1].set_ylabel('Recency (days)')
        axes[0, 1].set_title('Recency by Risk Category')
        
        # Frequency comparison
        axes[1, 0].boxplot([data_with_labels[data_with_labels['is_high_risk'] == 0]['frequency'],
                          data_with_labels[data_with_labels['is_high_risk'] == 1]['frequency']],
                          labels=['Not High-Risk', 'High-Risk'])
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Frequency by Risk Category')
        
        # Monetary comparison
        axes[1, 1].boxplot([data_with_labels[data_with_labels['is_high_risk'] == 0]['monetary'],
                          data_with_labels[data_with_labels['is_high_risk'] == 1]['monetary']],
                          labels=['Not High-Risk', 'High-Risk'])
        axes[1, 1].set_ylabel('Monetary')
        axes[1, 1].set_title('Monetary by Risk Category')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'risk_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def merge_with_original_data(self, original_data: pd.DataFrame, 
                               high_risk_labels: pd.Series,
                               customer_col: str = 'CustomerId') -> pd.DataFrame:
        """
        Merge high-risk labels back into original data
        
        Parameters:
        -----------
        original_data : pd.DataFrame
            Original transaction data
        high_risk_labels : pd.Series
            High-risk labels for each customer
        customer_col : str
            Customer identifier column
            
        Returns:
        --------
        DataFrame with high-risk labels merged
        """
        logger.info("Merging high-risk labels with original data...")
        
        # Create mapping from customer to risk label
        risk_mapping = high_risk_labels.to_dict()
        
        # Add risk label to original data
        data_with_risk = original_data.copy()
        data_with_risk['is_high_risk'] = data_with_risk[customer_col].map(risk_mapping)
        
        # Check for any customers without labels
        missing_labels = data_with_risk['is_high_risk'].isna().sum()
        if missing_labels > 0:
            logger.warning(f"{missing_labels} transactions have customers without risk labels")
            data_with_risk['is_high_risk'] = data_with_risk['is_high_risk'].fillna(0)  # Assume not high-risk
        
        logger.info(f"Added risk labels to {len(data_with_risk)} transactions")
        logger.info(f"Risk distribution in full dataset:")
        logger.info(data_with_risk['is_high_risk'].value_counts())
        
        return data_with_risk


def main():
    """Main function to demonstrate target variable engineering"""
    import sys
    import os
    
    # Add parent directory to path
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    # Example usage
    try:
        # Load processed data
        processed_data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'features_engineered.csv')
        logger.info(f"Loading processed data from {processed_data_path}")
        
        if not os.path.exists(processed_data_path):
            logger.error(f"Processed data file not found: {processed_data_path}")
            data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
            if os.path.exists(data_dir):
                logger.info(f"Files in {data_dir}: {os.listdir(data_dir)}")
            else:
                logger.error(f"Directory doesn't exist: {data_dir}")
        else:
            data = pd.read_csv(processed_data_path, nrows=1000)
            logger.info(f"Successfully loaded {len(data)} rows")    
        
        
        # Initialize target engineer
        target_engineer = TargetVariableEngineer(
            n_clusters=3,
            clustering_method='kmeans',
            use_multiple_criteria=True,
            risk_threshold=0.3,
            random_state=42
        )
        
        # Create proxy target variable
        rfm_with_labels, high_risk_labels = target_engineer.fit_transform(data)
        
        # Save RFM data with labels
        rfm_output_path = os.path.join(PROCESSED_DATA_DIR, 'rfm_with_labels.csv')
        rfm_with_labels.to_csv(rfm_output_path, index=True)
        logger.info(f"Saved RFM data with labels to {rfm_output_path}")
        
        # Merge labels back to original data
        data_with_risk_labels = target_engineer.merge_with_original_data(data, high_risk_labels)
        
        # Save final dataset with risk labels
        final_output_path = os.path.join(PROCESSED_DATA_DIR, 'data_with_risk_labels.csv')
        data_with_risk_labels.to_csv(final_output_path, index=False)
        logger.info(f"Saved final dataset with risk labels to {final_output_path}")
        
        # Create visualizations
        target_engineer.visualize_results(rfm_with_labels, REPORTS_DIR)
        logger.info("Target engineering completed successfully!")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        logger.info(f"Total customers analyzed: {len(rfm_with_labels)}")
        logger.info(f"High-risk customers identified: {high_risk_labels.sum()}")
        logger.info(f"High-risk percentage: {(high_risk_labels.sum() / len(high_risk_labels) * 100):.2f}%")
        
    except Exception as e:
        logger.error(f"Error in target variable engineering: {str(e)}")
        raise


if __name__ == "__main__":
    main()