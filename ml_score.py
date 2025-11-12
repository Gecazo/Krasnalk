"""
Machine Learning Scoring Module for Wrocław Walkability Analyzer
================================================================

Trains a Random Forest model to predict walkability scores based on
engineered features.

Features:
- Random Forest Regression with hyperparameter tuning
- Synthetic label generation via weighted formula
- Model persistence and evaluation metrics
- SHAP-based interpretability
- Feature importance visualization
"""

import sys
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Import project config
from config import PROCESSED_DATA_DIR, OUTPUT_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model parameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

# Feature weights for synthetic labels
FEATURE_WEIGHTS = {
    'sidewalk_density_m_per_km2': 0.20,
    'crosswalk_density_per_km2': 0.15,
    'amenity_count_1km': 0.15,
    'avg_amenity_distance_m': -0.10,  # Lower is better
    'transit_count_500m': 0.15,
    'avg_transit_distance_m': -0.10,  # Lower is better
    'network_connectivity': 0.15,
    'area_km2': 0.05,
    'min_amenity_distance_m': -0.05,  # Lower is better
    'min_transit_distance_m': -0.05   # Lower is better
}


class WalkabilityScorer:
    """ML model for predicting walkability scores."""
    
    def __init__(self, model_params: Dict = None):
        """
        Initialize the scorer.
        
        Args:
            model_params: RandomForest hyperparameters (uses RF_PARAMS if None)
        """
        self.params = model_params or RF_PARAMS
        self.model = RandomForestRegressor(**self.params)
        self.feature_columns = None
        self.scaler_params = {}
        
        logger.info(f"Initialized WalkabilityScorer with params: {self.params}")
    
    def generate_synthetic_labels(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Generate synthetic walkability scores using weighted formula.
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            Array of synthetic scores (0-100)
        """
        logger.info("Generating synthetic labels from feature weights...")
        
        # Normalize features to 0-1 scale
        normalized = features_df.copy()
        
        for col in features_df.columns:
            if col in ['neighborhood', 'centroid_lat', 'centroid_lon']:
                continue
            
            min_val = features_df[col].min()
            max_val = features_df[col].max()
            
            if max_val > min_val:
                normalized[col] = (features_df[col] - min_val) / (max_val - min_val)
            else:
                normalized[col] = 0.5
            
            # Store scaling parameters
            self.scaler_params[col] = {'min': min_val, 'max': max_val}
        
        # Calculate weighted score
        scores = np.zeros(len(features_df))
        
        for feature, weight in FEATURE_WEIGHTS.items():
            if feature in normalized.columns:
                if weight < 0:  # Invert negative features
                    scores += abs(weight) * (1 - normalized[feature])
                else:
                    scores += weight * normalized[feature]
        
        # Scale to 0-100 with some noise for realism
        scores = scores * 100
        noise = np.random.normal(0, 3, len(scores))  # Add small noise
        scores = np.clip(scores + noise, 0, 100)
        
        logger.info(f"Generated {len(scores)} synthetic labels")
        logger.info(f"Score range: {scores.min():.1f} - {scores.max():.1f}")
        logger.info(f"Mean score: {scores.mean():.1f}")
        
        return scores
    
    def prepare_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training/prediction.
        
        Args:
            features_df: Raw features DataFrame
            
        Returns:
            Cleaned features DataFrame
        """
        # Select numeric features only
        exclude_cols = ['neighborhood', 'centroid_lat', 'centroid_lon']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        X = features_df[feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        
        self.feature_columns = X.columns.tolist()
        logger.info(f"Prepared {len(self.feature_columns)} features: {self.feature_columns}")
        
        return X
    
    def train(self, features_df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Train the Random Forest model.
        
        Args:
            features_df: DataFrame with engineered features
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("=" * 60)
        logger.info("Starting ML Training Pipeline")
        logger.info("=" * 60)
        
        # Generate synthetic labels
        y = self.generate_synthetic_labels(features_df)
        
        # Prepare features
        X = self.prepare_features(features_df)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train model
        logger.info("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Evaluate
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
        }
        
        # Cross-validation
        logger.info("Running 5-fold cross-validation...")
        cv_scores = cross_val_score(
            self.model, X, y, cv=5, scoring='r2', n_jobs=-1
        )
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        # Log results
        logger.info("=" * 60)
        logger.info("Training Results:")
        logger.info(f"  Train R²: {metrics['train_r2']:.4f}")
        logger.info(f"  Test R²: {metrics['test_r2']:.4f}")
        logger.info(f"  Train RMSE: {metrics['train_rmse']:.2f}")
        logger.info(f"  Test RMSE: {metrics['test_rmse']:.2f}")
        logger.info(f"  Train MAE: {metrics['train_mae']:.2f}")
        logger.info(f"  Test MAE: {metrics['test_mae']:.2f}")
        logger.info(f"  CV R² (5-fold): {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
        logger.info("=" * 60)
        
        return metrics
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict walkability scores.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Array of predicted scores (0-100)
        """
        X = self.prepare_features(features_df)
        scores = self.model.predict(X)
        return np.clip(scores, 0, 100)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Returns:
            DataFrame with features and importance scores
        """
        if self.feature_columns is None:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def calculate_shap_values(self, features_df: pd.DataFrame, max_samples: int = 100):
        """
        Calculate SHAP values for model interpretability.
        
        Args:
            features_df: DataFrame with features
            max_samples: Maximum samples for background data (for performance)
            
        Returns:
            Tuple of (shap_values, explainer, X_data)
        """
        if self.feature_columns is None:
            raise ValueError("Model not trained yet")
        
        logger.info("Calculating SHAP values for local interpretability...")
        
        # Prepare features
        X = self.prepare_features(features_df)
        
        # Use a subset for background if dataset is large
        if len(X) > max_samples:
            background = X.sample(n=max_samples, random_state=42)
        else:
            background = X
        
        # Create TreeExplainer (optimized for tree-based models)
        explainer = shap.TreeExplainer(self.model, background)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        logger.info(f"SHAP values calculated for {len(X)} samples")
        
        return shap_values, explainer, X
    
    def plot_shap_summary(self, shap_values, X_data, output_path: str = 'outputs/shap_summary.png'):
        """
        Create SHAP summary plot showing feature importance and direction.
        
        Args:
            shap_values: SHAP values from calculate_shap_values
            X_data: Feature data
            output_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        shap.summary_plot(
            shap_values, 
            X_data, 
            feature_names=self.feature_columns,
            show=False,
            plot_size=(10, 8)
        )
        
        plt.title('SHAP Summary Plot - Feature Impact on Walkability Score', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP summary plot saved to {output_path}")
    
    def plot_shap_waterfall(self, shap_values, X_data, sample_idx: int = 0, 
                           neighborhood_name: str = None, 
                           output_path: str = 'outputs/shap_waterfall.png'):
        """
        Create SHAP waterfall plot for individual prediction explanation.
        
        Args:
            shap_values: SHAP values from calculate_shap_values
            X_data: Feature data
            sample_idx: Index of sample to explain
            neighborhood_name: Name of neighborhood (for title)
            output_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        # Create explanation object
        base_value = self.model.predict(X_data)[sample_idx]
        
        # Get SHAP values for this sample
        sample_shap = shap_values[sample_idx]
        sample_features = X_data.iloc[sample_idx]
        
        # Create waterfall data
        feature_names = self.feature_columns
        shap_dict = dict(zip(feature_names, sample_shap))
        feature_dict = dict(zip(feature_names, sample_features))
        
        # Sort by absolute SHAP value
        sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        
        # Create horizontal bar plot
        features = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]
        colors = ['red' if v < 0 else 'green' for v in values]
        
        plt.barh(features, values, color=colors, alpha=0.7)
        plt.xlabel('SHAP Value (impact on score)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        
        title = f'SHAP Explanation - Top 10 Features'
        if neighborhood_name:
            title += f'\nNeighborhood: {neighborhood_name}'
        plt.title(title, fontsize=14, fontweight='bold')
        
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP waterfall plot saved to {output_path}")
        
        return sorted_features
    
    def save_model(self, filepath: str = 'models/walkability_model.pkl'):
        """Save trained model to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'scaler_params': self.scaler_params,
            'params': self.params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str = 'models/walkability_model.pkl'):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        scorer = cls(model_params=model_data['params'])
        scorer.model = model_data['model']
        scorer.feature_columns = model_data['feature_columns']
        scorer.scaler_params = model_data['scaler_params']
        
        logger.info(f"Model loaded from {filepath}")
        return scorer


def plot_feature_importance(importance_df: pd.DataFrame, output_path: str = 'outputs/feature_importance.png'):
    """
    Plot feature importance bar chart.
    
    Args:
        importance_df: DataFrame with feature importance
        output_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    sns.barplot(
        data=importance_df.head(10),
        x='importance',
        y='feature',
        palette='viridis'
    )
    
    plt.title('Top 10 Feature Importance - Random Forest', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature importance plot saved to {output_path}")


def main():
    """Main execution function."""
    logger.info("Wrocław Walkability ML Scoring Module")
    logger.info("=" * 60)
    
    # Load features
    features_path = Path(PROCESSED_DATA_DIR) / 'neighborhood_features.csv'
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        logger.error("Please run data_gather.py first!")
        return
    
    logger.info(f"Loading features from {features_path}")
    features_df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(features_df)} neighborhoods with {len(features_df.columns)} features")
    
    # Initialize scorer
    scorer = WalkabilityScorer()
    
    # Train model
    metrics = scorer.train(features_df)
    
    # Get predictions for all neighborhoods
    scores = scorer.predict(features_df)
    
    # Create output DataFrame
    output_df = features_df[['neighborhood', 'centroid_lat', 'centroid_lon', 'area_km2']].copy()
    output_df['walkability_score'] = scores
    output_df['score_category'] = pd.cut(
        scores,
        bins=[0, 25, 50, 75, 100],
        labels=['Low', 'Medium', 'High', 'Excellent']
    )
    
    # Save scores
    output_path = Path(PROCESSED_DATA_DIR) / 'neighborhood_scores.csv'
    output_df.to_csv(output_path, index=False)
    logger.info(f"Saved scores to {output_path}")
    
    # Feature importance
    importance_df = scorer.get_feature_importance()
    logger.info("\nTop 10 Feature Importance:")
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Plot feature importance
    plot_feature_importance(importance_df)
    
    # SHAP Analysis
    logger.info("\n" + "=" * 60)
    logger.info("SHAP Analysis - Local Interpretability")
    logger.info("=" * 60)
    
    try:
        # Calculate SHAP values
        shap_values, explainer, X_data = scorer.calculate_shap_values(features_df)
        
        # Create SHAP summary plot
        scorer.plot_shap_summary(shap_values, X_data)
        
        # Create waterfall plot for highest and lowest scoring neighborhoods
        scores_with_idx = list(enumerate(scores))
        highest_idx = max(scores_with_idx, key=lambda x: x[1])[0]
        lowest_idx = min(scores_with_idx, key=lambda x: x[1])[0]
        
        # Highest scoring neighborhood
        highest_name = features_df.iloc[highest_idx]['neighborhood']
        logger.info(f"\nCreating SHAP explanation for HIGHEST scorer: {highest_name}")
        high_features = scorer.plot_shap_waterfall(
            shap_values, X_data, highest_idx, highest_name,
            'outputs/shap_waterfall_highest.png'
        )
        logger.info(f"Top 3 positive contributors for {highest_name}:")
        for feat, val in high_features[:3]:
            logger.info(f"  {feat}: {val:+.3f}")
        
        # Lowest scoring neighborhood
        lowest_name = features_df.iloc[lowest_idx]['neighborhood']
        logger.info(f"\nCreating SHAP explanation for LOWEST scorer: {lowest_name}")
        low_features = scorer.plot_shap_waterfall(
            shap_values, X_data, lowest_idx, lowest_name,
            'outputs/shap_waterfall_lowest.png'
        )
        logger.info(f"Top 3 negative contributors for {lowest_name}:")
        for feat, val in low_features[:3]:
            logger.info(f"  {feat}: {val:+.3f}")
        
        logger.info("\n✓ SHAP analysis complete!")
        
    except Exception as e:
        logger.warning(f"SHAP analysis failed (non-critical): {e}")
        logger.info("Continuing without SHAP plots...")
    
    # Save model
    scorer.save_model()
    
    logger.info("=" * 60)
    logger.info("ML Pipeline Complete!")
    logger.info(f"Model: models/walkability_model.pkl")
    logger.info(f"Scores: {output_path}")
    logger.info(f"Plots:")
    logger.info(f"  - Feature Importance: outputs/feature_importance.png")
    logger.info(f"  - SHAP Summary: outputs/shap_summary.png")
    logger.info(f"  - SHAP Highest: outputs/shap_waterfall_highest.png")
    logger.info(f"  - SHAP Lowest: outputs/shap_waterfall_lowest.png")
    logger.info("=" * 60)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("WALKABILITY SCORE SUMMARY")
    print("=" * 60)
    print(f"Total Neighborhoods: {len(output_df)}")
    print(f"\nScore Statistics:")
    print(f"  Mean: {scores.mean():.1f}")
    print(f"  Median: {np.median(scores):.1f}")
    print(f"  Min: {scores.min():.1f}")
    print(f"  Max: {scores.max():.1f}")
    print(f"  Std Dev: {scores.std():.1f}")
    print(f"\nScore Distribution:")
    print(output_df['score_category'].value_counts().sort_index())
    print("=" * 60)


if __name__ == "__main__":
    main()
