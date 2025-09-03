#!/usr/bin/env python3
"""
Test Original Format Data

This script validates that the generated data matches the exact requirements
of the original Aggressive_Drive_detection.ipynb notebook.
"""

import pandas as pd
import numpy as np
import os
import glob


def test_column_names():
    """Test that column names match exactly."""
    print("Testing column names...")
    
    data_dir = './original_format_data/'
    test_file = os.path.join(data_dir, '10.csv')
    
    if not os.path.exists(test_file):
        print("  ✗ Test file not found!")
        return False
    
    df = pd.read_csv(test_file)
    expected_columns = ['Car_ID', 'Time', 'Car_Orientation', 'Pitch_Rate', 'Roll_Rate', 
                       'Acceleration', 'Velocity', 'Steering_Wheel_Angle', 'Yaw_Rate']
    
    actual_columns = list(df.columns)
    
    if actual_columns == expected_columns:
        print(f"  ✓ Column names match exactly: {actual_columns}")
        return True
    else:
        print(f"  ✗ Column names mismatch!")
        print(f"    Expected: {expected_columns}")
        print(f"    Actual:   {actual_columns}")
        return False


def test_time_resets():
    """Test that Time column has resets to 0 (matching original notebook behavior)."""
    print("\nTesting time resets...")
    
    data_dir = './original_format_data/'
    test_file = os.path.join(data_dir, '10.csv')
    
    df = pd.read_csv(test_file)
    
    # Find all Time==0 points (like the original notebook does)
    time_zero_indices = df[df['Time'] == 0].index.tolist()
    
    if len(time_zero_indices) > 1:
        print(f"  ✓ Found {len(time_zero_indices)} Time==0 points: {time_zero_indices}")
        
        # Check that time increments properly between resets
        for i in range(1, len(time_zero_indices)):
            start_idx = time_zero_indices[i-1]
            end_idx = time_zero_indices[i]
            
            segment_length = end_idx - start_idx
            expected_time = np.arange(segment_length)
            actual_time = df.iloc[start_idx:end_idx]['Time'].values
            
            if np.array_equal(expected_time, actual_time):
                print(f"    ✓ Segment {i}: Time increments correctly (0 to {segment_length-1})")
            else:
                print(f"    ✗ Segment {i}: Time increment mismatch")
                return False
        
        return True
    else:
        print(f"  ⚠ Only found {len(time_zero_indices)} Time==0 point(s)")
        return True


def test_data_types():
    """Test that data types are appropriate."""
    print("\nTesting data types...")
    
    data_dir = './original_format_data/'
    test_file = os.path.join(data_dir, '10.csv')
    
    df = pd.read_csv(test_file)
    
    # Check data types
    expected_types = {
        'Car_ID': 'int64',
        'Time': 'int64',
        'Car_Orientation': 'float64',
        'Pitch_Rate': 'float64',
        'Roll_Rate': 'float64',
        'Acceleration': 'float64',
        'Velocity': 'float64',
        'Steering_Wheel_Angle': 'float64',
        'Yaw_Rate': 'float64'
    }
    
    all_correct = True
    for col, expected_type in expected_types.items():
        actual_type = str(df[col].dtype)
        if actual_type == expected_type:
            print(f"  ✓ {col}: {actual_type}")
        else:
            print(f"  ✗ {col}: expected {expected_type}, got {actual_type}")
            all_correct = False
    
    return all_correct


def test_value_ranges():
    """Test that values are within realistic ranges."""
    print("\nTesting value ranges...")
    
    data_dir = './original_format_data/'
    test_file = os.path.join(data_dir, '10.csv')
    
    df = pd.read_csv(test_file)
    
    # Define expected ranges
    expected_ranges = {
        'Car_ID': (10, 89),  # Driver IDs
        'Time': (0, 8000),   # Time indices
        'Car_Orientation': (0, 360),  # Degrees
        'Pitch_Rate': (-20, 20),      # Degrees/second
        'Roll_Rate': (-15, 15),       # Degrees/second
        'Acceleration': (-15, 15),    # m/s²
        'Velocity': (0, 130),         # km/h
        'Steering_Wheel_Angle': (-45, 45),  # Degrees
        'Yaw_Rate': (-25, 25)         # Degrees/second
    }
    
    all_correct = True
    for col, (min_val, max_val) in expected_ranges.items():
        actual_min = df[col].min()
        actual_max = df[col].max()
        
        if min_val <= actual_min <= actual_max <= max_val:
            print(f"  ✓ {col}: {actual_min:.2f} to {actual_max:.2f} (within {min_val} to {max_val})")
        else:
            print(f"  ✗ {col}: {actual_min:.2f} to {actual_max:.2f} (outside {min_val} to {max_val})")
            all_correct = False
    
    return all_correct


def test_segmented_files():
    """Test that segmented files are created correctly."""
    print("\nTesting segmented files...")
    
    data_dir = './original_format_data/'
    
    # Count main files
    main_files = glob.glob(os.path.join(data_dir, '*.csv'))
    main_files = [f for f in main_files if not f.endswith('OrderRight_') and not f.endswith('LessLength_')]
    
    # Count segmented files
    segmented_files = glob.glob(os.path.join(data_dir, 'OrderRight_*.csv'))
    
    print(f"  ✓ Main files: {len(main_files)}")
    print(f"  ✓ Segmented files: {len(segmented_files)}")
    
    # Test a few segmented files
    if len(segmented_files) > 0:
        test_segment = segmented_files[0]
        df_segment = pd.read_csv(test_segment)
        
        print(f"  ✓ Sample segment: {os.path.basename(test_segment)}")
        print(f"    Shape: {df_segment.shape}")
        print(f"    Time range: {df_segment['Time'].min()} to {df_segment['Time'].max()}")
        
        # Check that time starts from 0 in segments
        if df_segment['Time'].min() == 0:
            print(f"    ✓ Time starts from 0")
        else:
            print(f"    ✗ Time does not start from 0")
            return False
        
        return True
    else:
        print("  ⚠ No segmented files found")
        return True


def test_original_notebook_compatibility():
    """Test that data can be processed like the original notebook."""
    print("\nTesting original notebook compatibility...")
    
    data_dir = './original_format_data/'
    test_file = os.path.join(data_dir, '10.csv')
    
    # Simulate the original notebook's data loading process
    try:
        # Load data (like the original notebook)
        data = pd.read_csv(test_file, index_col=None, low_memory=False)
        
        # Check Car ID set (like the original notebook)
        car_ids = set(data['Car_ID'])
        print(f"  ✓ Driving Car ID Set: {car_ids}")
        
        # Check Time==0 points (like the original notebook)
        time_zero_count = len(data[data['Time'] == 0])
        print(f"  ✓ Time==0 count: {time_zero_count}")
        
        # Reset index (like the original notebook)
        data = data.reset_index().drop(['index'], axis=1)
        
        # Rename columns (like the original notebook)
        data.columns = ['Car_ID', 'Time', 'Car_Orientation', 'Pitch_Rate', 'Roll_Rate', 
                       'Acceleration', 'Velocity', 'Steering_Wheel_Angle', 'Yaw_Rate']
        
        print(f"  ✓ Data shape after processing: {data.shape}")
        print(f"  ✓ Columns after renaming: {list(data.columns)}")
        
        # Find separation points (like the original notebook)
        sep_list = list(data[data['Time'] == 0].index)
        
        if len(sep_list) == 0:
            sep_list = [0, len(data)]
        elif len(sep_list) > 0 and sep_list[0] > 0:
            if sep_list[-1] == len(data) - 1:
                sep_list = [0] + sep_list + [len(data) + 1]
            else:
                sep_list = [0] + sep_list + [len(data)]
        
        print(f"  ✓ Separation points: {sep_list}")
        
        # Check segment lengths
        for i in range(1, len(sep_list)):
            start_idx = sep_list[i-1]
            end_idx = sep_list[i]
            segment_length = end_idx - start_idx
            print(f"    Segment {i}: {segment_length} samples")
            
            if segment_length > 151:
                print(f"      ✓ Segment {i} is longer than 151 samples")
            else:
                print(f"      ⚠ Segment {i} is shorter than 151 samples")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error during notebook simulation: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("ORIGINAL FORMAT DATA VALIDATION")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Column Names", test_column_names),
        ("Time Resets", test_time_resets),
        ("Data Types", test_data_types),
        ("Value Ranges", test_value_ranges),
        ("Segmented Files", test_segmented_files),
        ("Notebook Compatibility", test_original_notebook_compatibility)
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
        print("🎉 All tests passed! Data is compatible with original notebook.")
        print("\nYou can now use this data with your Aggressive_Drive_detection.ipynb notebook!")
    else:
        print("⚠ Some tests failed. Please check the output above.")
    
    return passed == total


if __name__ == "__main__":
    main()
