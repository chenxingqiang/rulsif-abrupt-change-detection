#!/usr/bin/env python3
"""
Test Synthetic Driving Data

This script validates the generated synthetic driving data and demonstrates
how to use it with the anomaly detection library.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import anomaly_detection
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from anomaly_detection.time_series import RULSIFDetector
    from anomaly_detection.core.metrics import AnomalyMetrics
    from anomaly_detection.core.visualization import AnomalyVisualizer
    RULSIF_AVAILABLE = True
except ImportError:
    print("Warning: anomaly_detection package not available. Running basic tests only.")
    RULSIF_AVAILABLE = False


def test_data_structure():
    """Test the structure of generated data files."""
    print("Testing data structure...")
    
    data_dir = './synthetic_driving_data/'
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found!")
        return False
    
    # Check if files exist
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files")
    
    if len(csv_files) == 0:
        print("Error: No CSV files found!")
        return False
    
    # Test first file
    test_file = os.path.join(data_dir, csv_files[0])
    print(f"Testing file: {test_file}")
    
    try:
        df = pd.read_csv(test_file)
        print(f"  ✓ File loaded successfully")
        print(f"  ✓ Shape: {df.shape}")
        print(f"  ✓ Columns: {list(df.columns)}")
        
        # Check data types
        print(f"  ✓ Data types:")
        for col, dtype in df.dtypes.items():
            print(f"    {col}: {dtype}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() == 0:
            print(f"  ✓ No missing values")
        else:
            print(f"  ⚠ Missing values: {missing_values[missing_values > 0].to_dict()}")
        
        # Check value ranges
        print(f"  ✓ Value ranges:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'driver_id' and col != 'gear':
                min_val = df[col].min()
                max_val = df[col].min()
                print(f"    {col}: {min_val:.2f} to {max_val:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading file: {e}")
        return False


def test_data_quality():
    """Test the quality of generated data."""
    print("\nTesting data quality...")
    
    data_dir = './synthetic_driving_data/'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # Test multiple files
    all_stats = []
    for i, csv_file in enumerate(csv_files[:5]):  # Test first 5 files
        file_path = os.path.join(data_dir, csv_file)
        df = pd.read_csv(file_path)
        
        # Basic statistics
        stats = {
            'file': csv_file,
            'rows': len(df),
            'aggressive_events': df['aggressive_event'].sum(),
            'event_rate': df['aggressive_event'].mean(),
            'speed_mean': df['speed'].mean(),
            'speed_std': df['speed'].std(),
            'rpm_mean': df['rpm'].mean(),
            'rpm_std': df['rpm'].std()
        }
        all_stats.append(stats)
        
        print(f"  {csv_file}: {len(df)} rows, {stats['aggressive_events']} events ({stats['event_rate']:.1%})")
    
    # Check consistency
    row_counts = [s['rows'] for s in all_stats]
    if len(set(row_counts)) == 1:
        print(f"  ✓ All files have consistent row count: {row_counts[0]}")
    else:
        print(f"  ⚠ Inconsistent row counts: {row_counts}")
    
    # Check event rates
    event_rates = [s['event_rate'] for s in all_stats]
    avg_event_rate = np.mean(event_rates)
    print(f"  ✓ Average event rate: {avg_event_rate:.1%}")
    
    return True


def test_anomaly_detection():
    """Test anomaly detection on synthetic data."""
    if not RULSIF_AVAILABLE:
        print("\nSkipping anomaly detection test (package not available)")
        return False
    
    print("\nTesting anomaly detection...")
    
    try:
        # Load data
        data_dir = './synthetic_driving_data/'
        test_file = os.path.join(data_dir, '10.csv')  # Use driver 10
        df = pd.read_csv(test_file)
        
        # Prepare features
        feature_cols = ['speed', 'acceleration', 'rpm', 'fuel_consumption', 
                       'engine_temperature', 'brake_pressure', 'steering_angle', 
                       'lateral_acceleration', 'throttle_position']
        X = df[feature_cols].values
        
        # Initialize RULSIF detector
        detector = RULSIFDetector(
            alpha=0.5,
            n_kernels=30,
            n_folds=3,
            random_state=42
        )
        
        # Split data
        split_point = len(X) // 2
        reference_data = X[:split_point]
        test_data = X[split_point:]
        
        print(f"  Training data shape: {reference_data.shape}")
        print(f"  Test data shape: {test_data.shape}")
        
        # Fit detector
        print("  Fitting RULSIF detector...")
        detector.fit(X, reference_data=reference_data, test_data=test_data)
        
        print(f"  ✓ Optimal sigma: {detector.sigma_:.4f}")
        print(f"  ✓ Optimal lambda: {detector.lambda_:.4f}")
        
        # Detect anomalies
        print("  Detecting anomalies...")
        anomaly_scores = detector.score_samples(X)
        detected_anomalies = detector.detect_changes(X)
        
        print(f"  ✓ Anomaly scores shape: {anomaly_scores.shape}")
        print(f"  ✓ Detected anomalies: {np.sum(detected_anomalies)}")
        
        # Evaluate performance
        y_true = df['aggressive_event'].values
        y_pred = detected_anomalies
        
        # Basic metrics
        accuracy = np.mean(y_true == y_pred)
        precision = np.sum((y_true == 1) & (y_pred == 1)) / max(np.sum(y_pred == 1), 1)
        recall = np.sum((y_true == 1) & (y_pred == 1)) / max(np.sum(y_true == 1), 1)
        
        print(f"  ✓ Detection accuracy: {accuracy:.3f}")
        print(f"  ✓ Precision: {precision:.3f}")
        print(f"  ✓ Recall: {recall:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error in anomaly detection: {e}")
        return False


def create_sample_visualization():
    """Create a sample visualization of the data."""
    print("\nCreating sample visualization...")
    
    try:
        # Load data
        data_dir = './synthetic_driving_data/'
        test_file = os.path.join(data_dir, '10.csv')
        df = pd.read_csv(test_file)
        
        # Create plot
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Sample Synthetic Driving Data (Driver 10)', fontsize=16)
        
        # Speed and acceleration
        axes[0, 0].plot(df['speed'][:200], label='Speed (km/h)', color='blue')
        axes[0, 0].set_title('Speed Over Time')
        axes[0, 0].set_ylabel('Speed (km/h)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(df['acceleration'][:200], label='Acceleration (m/s²)', color='red')
        axes[0, 1].set_title('Acceleration Over Time')
        axes[0, 1].set_ylabel('Acceleration (m/s²)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # RPM and fuel consumption
        axes[1, 0].plot(df['rpm'][:200], label='RPM', color='green')
        axes[1, 0].set_title('Engine RPM Over Time')
        axes[1, 0].set_ylabel('RPM')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(df['fuel_consumption'][:200], label='Fuel (L/100km)', color='orange')
        axes[1, 1].set_title('Fuel Consumption Over Time')
        axes[1, 1].set_ylabel('Fuel (L/100km)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Steering and brake pressure
        axes[2, 0].plot(df['steering_angle'][:200], label='Steering (°)', color='purple')
        axes[2, 0].set_title('Steering Angle Over Time')
        axes[2, 0].set_ylabel('Steering Angle (°)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(df['brake_pressure'][:200], label='Brake (%)', color='brown')
        axes[2, 1].set_title('Brake Pressure Over Time')
        axes[2, 1].set_ylabel('Brake Pressure (%)')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # Mark aggressive events
        aggressive_indices = df[df['aggressive_event'] == True].index[:200]
        for ax in axes.flat:
            for idx in aggressive_indices:
                if idx < 200:
                    ax.axvline(x=idx, color='red', alpha=0.3, linewidth=0.5)
        
        plt.tight_layout()
        
        # Save plot
        output_file = './synthetic_driving_data/sample_visualization.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Visualization saved to: {output_file}")
        
        plt.show()
        return True
        
    except Exception as e:
        print(f"  ✗ Error creating visualization: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("SYNTHETIC DRIVING DATA VALIDATION")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Data Structure", test_data_structure),
        ("Data Quality", test_data_quality),
        ("Anomaly Detection", test_anomaly_detection),
        ("Visualization", create_sample_visualization)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name} Test:")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Synthetic data is ready to use.")
    else:
        print("⚠ Some tests failed. Please check the output above.")
    
    return passed == total


if __name__ == "__main__":
    main()
