#!/usr/bin/env python3
"""
PyOD Integration Example

This script demonstrates how to integrate PyOD (Python Outlier Detection)
with our Awesome Anomaly Detection framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Try to import PyOD
try:
    from pyod.utils.data import generate_data, evaluate_print
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF
    from pyod.models.iforest import IForest
    from pyod.models.ocsvm import OCSVM
    from pyod.models.copod import COPOD
    from pyod.utils.example import visualize
    PYOD_AVAILABLE = True
    print("✅ PyOD successfully imported!")
except ImportError:
    PYOD_AVAILABLE = False
    print("⚠️ PyOD not available. Install with: pip install pyod")
    print("   This example will show the structure but won't run the actual algorithms.")


class PyODIntegration:
    """Integration class for PyOD with our anomaly detection framework."""
    
    def __init__(self):
        """Initialize the PyOD integration."""
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def generate_synthetic_data(self, n_samples=1000, n_features=10, contamination=0.1):
        """Generate synthetic data for demonstration."""
        print(f"Generating synthetic data: {n_samples} samples, {n_features} features, {contamination*100}% contamination")
        
        if PYOD_AVAILABLE:
            # Use PyOD's data generation
            X_train, X_test, y_train, y_test = generate_data(
                n_train=n_samples//2, 
                n_test=n_samples//2, 
                contamination=contamination,
                n_features=n_features
            )
            
            # Combine train and test for our analysis
            X = np.vstack([X_train, X_test])
            y = np.hstack([y_train, y_test])
            
            # Add some noise and make it more realistic
            X += np.random.normal(0, 0.1, X.shape)
            
        else:
            # Fallback: generate our own synthetic data
            np.random.seed(42)
            X = np.random.randn(n_samples, n_features)
            y = np.zeros(n_samples)
            
            # Add some anomalies
            n_anomalies = int(n_samples * contamination)
            anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
            
            # Make anomalies by adding large values to some features
            for idx in anomaly_indices:
                feature_idx = np.random.randint(0, n_features)
                X[idx, feature_idx] += np.random.uniform(3, 5) * np.random.choice([-1, 1])
                y[idx] = 1
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def initialize_models(self):
        """Initialize various PyOD models."""
        if not PYOD_AVAILABLE:
            print("⚠️ PyOD not available, skipping model initialization")
            return
            
        print("Initializing PyOD models...")
        
        # 1. K-Nearest Neighbors (KNN)
        self.models['KNN'] = KNN(
            contamination=0.1,
            n_neighbors=5,
            method='largest'
        )
        
        # 2. Local Outlier Factor (LOF)
        self.models['LOF'] = LOF(
            contamination=0.1,
            n_neighbors=20,
            algorithm='auto'
        )
        
        # 3. Isolation Forest
        self.models['IsolationForest'] = IForest(
            contamination=0.1,
            n_estimators=100,
            random_state=42
        )
        
        # 4. One-Class SVM
        self.models['OCSVM'] = OCSVM(
            contamination=0.1,
            kernel='rbf',
            nu=0.1
        )
        
        # 5. COPOD (Copula-based Outlier Detection)
        self.models['COPOD'] = COPOD(
            contamination=0.1,
            random_state=42
        )
        
        print(f"✅ Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def train_and_evaluate_models(self, X, y):
        """Train all models and evaluate their performance."""
        if not PYOD_AVAILABLE:
            print("⚠️ PyOD not available, skipping training")
            return
            
        print("\n" + "="*60)
        print("TRAINING AND EVALUATING MODELS")
        print("="*60)
        
        # Split data for evaluation
        split_idx = len(X) // 2
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        for name, model in self.models.items():
            print(f"\n🔍 Training {name}...")
            
            try:
                # Train the model
                model.fit(X_train)
                
                # Get predictions and scores
                y_train_pred = model.labels_
                y_train_scores = model.decision_scores_
                
                y_test_pred = model.predict(X_test)
                y_test_scores = model.decision_function(X_test)
                
                # Store results
                self.results[name] = {
                    'train_pred': y_train_pred,
                    'train_scores': y_train_scores,
                    'test_pred': y_test_pred,
                    'test_scores': y_test_scores,
                    'model': model
                }
                
                # Evaluate performance
                print(f"  ✅ Training completed")
                print(f"  📊 Training set - Anomalies detected: {np.sum(y_train_pred)}/{len(y_train_pred)}")
                print(f"  📊 Test set - Anomalies detected: {np.sum(y_test_pred)}/{len(y_test_pred)}")
                
                # Calculate metrics
                if np.sum(y_test) > 0:  # Only if we have anomalies in test set
                    auc = roc_auc_score(y_test, y_test_scores)
                    print(f"  🎯 ROC AUC: {auc:.3f}")
                
            except Exception as e:
                print(f"  ❌ Error training {name}: {e}")
                continue
        
        print(f"\n✅ Training completed for {len(self.results)} models")
    
    def compare_models(self):
        """Compare the performance of different models."""
        if not self.results:
            print("⚠️ No results to compare")
            return
            
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_data = []
        
        for name, result in self.results.items():
            # Calculate metrics
            train_anomaly_rate = np.mean(result['train_pred'])
            test_anomaly_rate = np.mean(result['test_pred'])
            
            # Calculate score statistics
            train_scores = result['train_scores']
            test_scores = result['test_scores']
            
            comparison_data.append({
                'Model': name,
                'Train_Anomaly_Rate': f"{train_anomaly_rate:.3f}",
                'Test_Anomaly_Rate': f"{test_anomaly_rate:.3f}",
                'Train_Score_Mean': f"{np.mean(train_scores):.3f}",
                'Train_Score_Std': f"{np.std(train_scores):.3f}",
                'Test_Score_Mean': f"{np.mean(test_scores):.3f}",
                'Test_Score_Std': f"{np.std(test_scores):.3f}"
            })
        
        # Display comparison table
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def visualize_results(self, X, y):
        """Visualize the results of different models."""
        if not self.results:
            print("⚠️ No results to visualize")
            return
            
        print("\n" + "="*60)
        print("VISUALIZING RESULTS")
        print("="*60)
        
        # Create subplots for each model
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PyOD Models Comparison - Anomaly Detection Results', fontsize=16)
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(self.results.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Plot normal vs anomaly points
            normal_mask = y == 0
            anomaly_mask = y == 1
            
            # Use first two features for 2D visualization
            if X.shape[1] >= 2:
                ax.scatter(X[normal_mask, 0], X[normal_mask, 1], 
                          c='blue', alpha=0.6, s=20, label='Normal')
                ax.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1], 
                          c='red', alpha=0.8, s=30, label='Anomaly')
                
                # Highlight detected anomalies
                detected_anomalies = result['test_pred'] == 1
                if np.sum(detected_anomalies) > 0:
                    ax.scatter(X[split_idx:][detected_anomalies, 0], 
                              X[split_idx:][detected_anomalies, 1], 
                              c='orange', alpha=0.9, s=50, marker='x', 
                              label='Detected Anomalies')
                
                ax.set_xlabel('Feature 1')
                ax.set_ylabel('Feature 2')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                # 1D case
                ax.hist(X[normal_mask, 0], bins=30, alpha=0.7, label='Normal', color='blue')
                ax.hist(X[anomaly_mask, 0], bins=30, alpha=0.7, label='Anomaly', color='red')
                ax.set_xlabel('Feature 1')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            ax.set_title(f'{name}\nAnomalies: {np.sum(result["test_pred"])}/{len(result["test_pred"])}')
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Score distribution plots
        self._plot_score_distributions()
    
    def _plot_score_distributions(self):
        """Plot anomaly score distributions for different models."""
        if not self.results:
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Anomaly Score Distributions by Model', fontsize=16)
        
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(self.results.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Plot score distributions
            train_scores = result['train_scores']
            test_scores = result['test_scores']
            
            ax.hist(train_scores, bins=30, alpha=0.7, label='Training', color='blue')
            ax.hist(test_scores, bins=30, alpha=0.7, label='Test', color='red')
            
            # Add vertical line for threshold (if available)
            if hasattr(result['model'], 'threshold_'):
                ax.axvline(result['model'].threshold_, color='green', linestyle='--', 
                          label=f'Threshold: {result["model"].threshold_:.3f}')
            
            ax.set_xlabel('Anomaly Score')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name} - Score Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(self.results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_advanced_features(self):
        """Demonstrate advanced PyOD features."""
        if not PYOD_AVAILABLE:
            print("⚠️ PyOD not available, skipping advanced features")
            return
            
        print("\n" + "="*60)
        print("ADVANCED PYD FEATURES")
        print("="*60)
        
        # 1. Model combination
        print("🔧 Model Combination Example:")
        print("   PyOD supports combining multiple models for better performance")
        
        # 2. Confidence scores
        print("🎯 Confidence Scores:")
        print("   Many PyOD models provide confidence scores with predictions")
        
        # 3. Online learning
        print("📈 Online Learning:")
        print("   Some models support incremental learning for streaming data")
        
        # 4. Custom thresholds
        print("⚖️ Custom Thresholds:")
        print("   You can set custom thresholds for anomaly detection")
        
        print("\nFor more advanced features, check the PyOD documentation!")


def main():
    """Main function to demonstrate PyOD integration."""
    print("🚀 PYOD INTEGRATION WITH AWESOME ANOMALY DETECTION")
    print("=" * 60)
    
    # Initialize integration
    integration = PyODIntegration()
    
    # Generate synthetic data
    print("\n📊 DATA GENERATION")
    print("-" * 40)
    X, y = integration.generate_synthetic_data(
        n_samples=1000, 
        n_features=10, 
        contamination=0.15
    )
    
    print(f"Data shape: {X.shape}")
    print(f"Anomaly rate: {np.mean(y):.3f} ({np.sum(y)} anomalies)")
    
    # Initialize models
    print("\n🤖 MODEL INITIALIZATION")
    print("-" * 40)
    integration.initialize_models()
    
    # Train and evaluate models
    integration.train_and_evaluate_models(X, y)
    
    # Compare models
    comparison_df = integration.compare_models()
    
    # Visualize results
    integration.visualize_results(X, y)
    
    # Demonstrate advanced features
    integration.demonstrate_advanced_features()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("✅ PyOD integration completed successfully!")
    print("📚 Key benefits of PyOD:")
    print("   - 50+ anomaly detection algorithms")
    print("   - Unified API similar to scikit-learn")
    print("   - High performance and scalability")
    print("   - Active community and documentation")
    print("\n🔗 Next steps:")
    print("   - Install PyOD: pip install pyod")
    print("   - Explore more algorithms")
    print("   - Integrate with your specific use cases")
    print("   - Check PyOD documentation for advanced features")


if __name__ == "__main__":
    main()
