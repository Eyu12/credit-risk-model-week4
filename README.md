# Credit Risk Modeling Project

## Task 1: Credit Scoring Business Understanding

### 1. Basel II Accord's Influence on Model Requirements

The Basel II Accord fundamentally changed how financial institutions approach credit risk by introducing three pillars: Minimum Capital Requirements, Supervisory Review, and Market Discipline. For our model development, this has critical implications:

**Impact on Model Interpretability and Documentation:**
- **Pillar 1 (Minimum Capital Requirements):** Requires banks to calculate regulatory capital based on credit risk. Our model must be transparent enough for regulators to understand how risk weights are derived.
- **Pillar 2 (Supervisory Review):** Regulators need to assess internal risk management processes. Our model documentation must clearly explain methodology, assumptions, and limitations.
- **Pillar 3 (Market Discipline):** Requires disclosure of risk exposures and risk assessment processes. Model transparency becomes a public accountability requirement.

**Why Interpretability Matters:**
- Regulatory compliance demands models whose decisions can be explained to both regulators and customers
- Auditors must be able to trace risk calculations back to underlying assumptions
- Model governance frameworks require clear documentation of development, validation, and monitoring processes

### 2. Proxy Variables in Credit Scoring

**Why Proxy Variables are Necessary:**
In our dataset, we lack explicit "default" labels because:
1. Transaction-level data doesn't directly indicate loan default
2. We need to infer credit risk from behavioral patterns
3. Direct default data may be unavailable due to privacy regulations or data collection limitations

**Potential Proxy Variables for Our Dataset:**
- **Fraudulent Transactions** (`FraudResult = 1`): May indicate higher risk behavior
- **Negative Amount Patterns**: Repeated fee/charge transactions might indicate financial stress
- **Transaction Frequency**: Unusual patterns could signal financial distress
- **Product Category Mix**: Certain product combinations may correlate with risk

**Business Risks of Proxy-Based Predictions:**
1. **Misclassification Risk**: Proxy variables may not perfectly correlate with actual credit risk
2. **Regulatory Risk**: Using non-standard risk indicators may face regulatory scrutiny
3. **Fairness Risk**: Proxies might inadvertently discriminate against certain groups
4. **Model Drift Risk**: Relationships between proxies and actual risk may change over time
5. **Validation Challenges**: Difficult to validate model accuracy without true default labels

### 3. Model Complexity Trade-offs in Financial Regulation

**Simple, Interpretable Models (Logistic Regression with WoE):**

*Advantages:*
- **Regulatory Compliance**: Easier to explain and validate for regulators
- **Transparency**: Clear feature importance through coefficients
- **Stability**: Less prone to overfitting with small datasets
- **Documentation**: Simpler to document and audit
- **Implementation**: Easier to deploy in production systems

*Disadvantages:*
- **Performance**: May not capture complex non-linear relationships
- **Feature Engineering**: Requires extensive manual feature engineering
- **Interaction Effects**: Limited ability to capture feature interactions automatically

**Complex, High-Performance Models (Gradient Boosting):**

*Advantages:*
- **Performance**: Often achieves higher predictive accuracy
- **Non-linear Patterns**: Can capture complex relationships automatically
- **Feature Interactions**: Automatically learns interactions between features
- **Robustness**: Better handles missing values and outliers

*Disadvantages:*
- **Black Box Nature**: Difficult to explain individual predictions
- **Regulatory Hurdles**: May face challenges in regulatory approval processes
- **Overfitting Risk**: More prone to overfitting without careful regularization
- **Implementation Complexity**: More difficult to deploy and monitor
- **Computational Cost**: Higher resource requirements for training and inference

**Recommended Approach for Our Context:**
Given the regulated financial environment and need for transparency, we recommend:
1. **Start with Interpretable Models**: Begin with Logistic Regression with WoE encoding
2. **Use Ensemble Methods Cautiously**: Consider Gradient Boosting only if significant performance gains justify the complexity
3. **Implement SHAP/LIME**: Use model-agnostic explainability tools if using complex models
4. **Maintain Comprehensive Documentation**: Regardless of model choice, document every step thoroughly
5. **Establish Model Governance**: Create clear processes for model validation, monitoring, and updates

**Conclusion:** In financial services, model interpretability isn't just a technical preference—it's a regulatory requirement. Our approach must balance predictive power with explainability, ensuring that our credit risk model can withstand regulatory scrutiny while providing meaningful risk assessments.

## Task 2:
###  Overview
This exploratory data analysis (EDA) notebook performs comprehensive analysis of a fraud detection dataset to identify patterns, anomalies, and insights that can inform fraud prediction models.

### 1. Data Overview
- Dataset dimensions and structure
- Column names and data types
- Initial data inspection

### 2. Missing Values Analysis
- Identification of missing data
- Percentage calculations for missing values
- Assessment of data completeness

### 3. Distribution Analysis
- Numerical Features: Histograms and statistics for Amount, Value, PricingStrategy, FraudResult
- Categorical Features: Bar plots for CurrencyCode, CountryCode, ProductCategory, ChannelId, ProviderId

### 4. Fraud Analysis
- Fraud vs Non-fraud transaction distribution
- Fraud rates by product category
- Fraud percentage calculations

### 5. Temporal Analysis
- Transaction volume by hour of day
- Fraud patterns across different hours
- Day-of-week analysis (implied)

### 6. Correlation Analysis
- Correlation matrix for key numerical features
- Heatmap visualization

### 7. Outlier Detection
- Box plots for Amount and Value
- IQR-based outlier identification
- Outlier statistics and boundaries

### 8. Customer Behavior Analysis
- Top customers by transaction frequency
- Average transaction amounts by customer
- Customer segmentation insights

### 9. Provider Analysis
- Transaction statistics by provider
- Fraud rates across different providers
- Provider risk assessment

### 10. Key Insights Summary
- Top 5 data-driven insights
- Patterns and anomalies identified
- Data quality assessment

## Task 3: Data Processing Module

This module provides comprehensive feature engineering for credit risk modeling.

## Features Implemented

### 1. DateTime Features
- Hour, day, month, year extraction
- Cyclical encoding for hour (sin/cos)
- Business hour detection
- Day of week, week of year

### 2. Customer Aggregate Features
- Total transaction amount per customer
- Average transaction amount per customer
- Transaction count per customer
- Standard deviation of transaction amounts
- Min, max, median amounts
- Transaction frequency
- Outlier detection

### 3. Provider & Product Features
- Provider fraud rate
- Product category fraud rate
- Provider-product risk interactions

### 4. Channel Features
- Channel fraud rate
- Channel transaction volume
- Amount vs channel average

### 5. Financial Pattern Features
- Amount/Value ratios
- Fee transaction detection
- Transaction size categorization
- Amount transformations (log, sqrt)

### 6. Negative Amount (Fee) Patterns
- Total fees per customer
- Average fee amount
- Fee frequency
- Fee-to-transaction ratio

### 7. Weight of Evidence (WoE) Transformation
- Automatic WoE encoding for categorical variables
- Information Value (IV) calculation
- IV-based feature importance

## Usage

python
from src.data_processing import DataProcessor

# Initialize processor
processor = DataProcessor(target_col='FraudResult')

# Process data
processed_data = processor.fit_transform(raw_data)

# Get feature importance
importance_df = processor.get_feature_importance()

## Task 4: Proxy Target Variable Engineering

## Overview

This module creates a proxy credit risk target variable using RFM (Recency, Frequency, Monetary) analysis and clustering. Since we don't have explicit default labels, we identify "disengaged" customers as high-risk proxies.

## Methodology

### 1. RFM Calculation
For each customer, we calculate:
- **Recency**: Days since last transaction (relative to snapshot date)
- **Frequency**: Total number of transactions
- **Monetary**: Total transaction amount (absolute value)
- **Additional Metrics**: Average transaction, consistency, activity score

### 2. Customer Segmentation
Using clustering algorithms (K-Means by default), we segment customers into groups based on their RFM profiles. The algorithm automatically identifies the least engaged cluster as high-risk.

### 3. High-Risk Labeling
Two approaches are implemented:
- **Single Cluster Method**: Label all customers in the identified high-risk cluster as high-risk
- **Multiple Criteria Method**: Combine clustering with additional risk factors using a weighted risk score

## Key Features

### RFMCalculator
- Calculates comprehensive RFM metrics
- Supports custom snapshot dates
- Creates RFM segments and scores
- Handles negative amounts (fees) appropriately

### CustomerClusterer
- Multiple clustering algorithms: K-Means, GMM, Agglomerative, DBSCAN
- Automatic high-risk cluster identification
- Clustering quality metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
- Comprehensive cluster analysis and visualization

### HighRiskLabeler
- Flexible labeling strategies
- Risk score calculation and threshold-based labeling
- Statistical analysis of high-risk vs not-high-risk groups
- Support for single or multiple risk criteria

### TargetVariableEngineer
- Complete pipeline integration
- Visualization of results
- Merge labels back to original transaction data
- Comprehensive reporting

## Usage

python
from src.target_engineering import TargetVariableEngineer

# Initialize the engineer
target_engineer = TargetVariableEngineer(
    n_clusters=3,
    clustering_method='kmeans',
    use_multiple_criteria=True,
    risk_threshold=0.3,
    random_state=42
)

# Create proxy target variable
rfm_with_labels, high_risk_labels = target_engineer.fit_transform(transaction_data)

# Merge labels back to original data
data_with_risk_labels = target_engineer.merge_with_original_data(
    transaction_data, 
    high_risk_labels
)

# Save results
data_with_risk_labels.to_csv('../data/processed/data_with_risk_labels.csv', index=False)

## Task 5: Model Training and Tracking

### Overview
This module implements a comprehensive model training pipeline with experiment tracking using MLflow. It trains multiple models, performs hyperparameter tuning, evaluates performance, and tracks all experiments.

### Features

#### 1. Data Preparation
- Automated train-test split with stratification
- Handling of missing values
- Feature preprocessing (scaling, encoding)
- Class imbalance analysis

#### 2. Model Training
- Multiple model architectures:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting (XGBoost, LightGBM)
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Cross-validation for robust evaluation

#### 3. Experiment Tracking with MLflow
- Logs all model parameters
- Tracks evaluation metrics
- Stores model artifacts
- Records confusion matrices and classification reports
- Model registry for versioning

#### 4. Model Evaluation Metrics
- Accuracy
- Precision
- Recall (Sensitivity)
- F1 Score
- ROC-AUC Score
- Average Precision Score

### Usage

python
from src.train import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(
    experiment_name="credit-risk-modeling",
    tracking_uri="../mlruns",
    random_state=42
)

# Run complete training pipeline
trainer.main()

## Task 6: Model Deployment and Continuous Integration

### Overview
This module provides a complete deployment solution for the credit risk prediction model, including:
- FastAPI REST API with comprehensive endpoints
- Docker containerization
- Docker Compose for multi-service orchestration
- CI/CD pipeline with GitHub Actions
- Code quality checks and automated testing

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API info |
| `/health` | GET | Health check and service status |
| `/model/info` | GET | Model information and version |
| `/predict` | POST | Predict risk for single transaction |
| `/predict/batch` | POST | Batch prediction for multiple transactions |
| `/features/example` | GET | Example feature structure |

### Request/Response Examples

 **Single Prediction Request:**
  ```json
  {
    "customer_id": "CustomerId_1234",
     "amount": 1000.0,
    "value": 1000.0,
    "product_category": "airtime",
    "provider_id": "ProviderId_6",
    "channel_id": "ChannelId_3",
    "transaction_hour": 14,
    "transaction_day": 15,
    "transaction_month": 11,
    "transaction_year": 2018
    }

## Project structure 
credit-risk-model/
├── .github/workflows/
│   └── ci.yml                    # CI/CD Pipeline
├── data/
│   ├── raw/                      # Raw data (gitignored)
│   └── processed/                # Processed data (gitignored)
├── notebooks/
│   └── eda.ipynb                 # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py        # Feature engineering
│   ├── train.py                  # Model training with MLflow
│   ├── predict.py                # Batch predictions
│   └── api/
│       ├── __init__.py
│       ├── main.py               # FastAPI application
│       └── pydantic_models.py    # Request/response models
├── tests/
│   ├── test_data_processing.py   # Unit tests
│   ├── test_target_variable.py   # Target engineering tests
│   └── test_api.py               # API tests
├── config/
│   └── training_config.py        # Training configuration
├── artifacts/                    # Model artifacts (gitignored)
├── logs/                         # Log files (gitignored)
├── mlruns/                       # MLflow runs (gitignored)
├── reports/                      # Reports and visualizations
├── scripts/
│   └── create_test_model.py      # Create test model
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Multi-service orchestration
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── README.md                     # Project documentation
└── mlflow.yaml                   # MLflow configuration