# test_ordinal_encoder_extension.py

import numpy as np
import pandas as pd
import pytest
import yaml
import sys
with open('../env_vars.yml', 'r') as file:
    config = yaml.safe_load(file)
# custom imports
sys.path.append(config['project_directory'])
from src.encoders import OrdinalEncoderExtensionUnknowns

@pytest.fixture
def setup_transformer():
    """Fixture to set up the custom transformer."""
    return OrdinalEncoderExtensionUnknowns()

def test_transform_numpy_array(setup_transformer):
    """Test transformation with a NumPy array."""
    input_array = np.array([1, 2, -1, 3])
    expected_output = np.array([2, 3, 0, 4])
    
    transformed_array = setup_transformer.transform(input_array)
    np.testing.assert_array_equal(transformed_array, expected_output)

def test_transform_series(setup_transformer):
    """Test transformation with a pandas Series."""
    input_series = pd.Series([1, 2, -1, 3])
    expected_output = pd.Series([2, 3, 0, 4])
    
    transformed_series = setup_transformer.transform(input_series)
    pd.testing.assert_series_equal(transformed_series, expected_output)

def test_transform_dataframe(setup_transformer):
    """Test transformation with a pandas DataFrame."""
    input_df = pd.DataFrame({
        'col1': [1, 2, -1],
        'col2': [-1, 3, 4]
    })
    expected_output_df = pd.DataFrame({
        'col1': [2, 3, 0],
        'col2': [0, 4, 5]
    })
    
    transformed_df = setup_transformer.transform(input_df)
    pd.testing.assert_frame_equal(transformed_df, expected_output_df)

def test_inverse_transform_numpy_array(setup_transformer):
    """Test inverse transformation with a NumPy array."""
    input_array = np.array([2, 3, 0, 4])
    expected_output = np.array([1, 2, -1, 3])
    
    inverse_array = setup_transformer.inverse_transform(input_array)
    np.testing.assert_array_equal(inverse_array, expected_output)

def test_inverse_transform_series(setup_transformer):
    """Test inverse transformation with a pandas Series."""
    input_series = pd.Series([2, 3, 0, 4])
    expected_output = pd.Series([1, 2, -1, 3])
    
    inverse_series = setup_transformer.inverse_transform(input_series)
    pd.testing.assert_series_equal(inverse_series, expected_output)

def test_inverse_transform_dataframe(setup_transformer):
    """Test inverse transformation with a pandas DataFrame."""
    input_df = pd.DataFrame({
        'col1': [2, 3, 0],
        'col2': [0, 4, 5]
    })
    expected_output_df = pd.DataFrame({
        'col1': [1, 2, -1],
        'col2': [-1, 3, 4]
    })
    
    inverse_df = setup_transformer.inverse_transform(input_df)
    pd.testing.assert_frame_equal(inverse_df, expected_output_df)

def test_transform_invalid_input(setup_transformer):
    """Test transformation with invalid input type."""
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame, Series, or NumPy array"):
        setup_transformer.transform("invalid input")

if __name__ == "__main__":
    pytest.main()
