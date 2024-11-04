import pytest
import numpy as np
from sklearn.metrics import confusion_matrix

# import custom functions
import yaml
import sys
with open('../env_vars.yml', 'r') as file:
    config = yaml.safe_load(file)
# custom imports
sys.path.append(config['project_directory'])
from src import utils


@pytest.fixture
def setup_data():
    # Using sklearn confusion_matrix
    y_true = ["cat", "dog", "bird", "cat", "bird"]
    y_pred = ["cat", "dog", "cat", "cat", "bird"]
    class_labels = np.sort(["bird", "dog", "cat"])  # the order in which classes appear in the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
    
    cost_matrix = [[1, -10], [-5, 1]]
    class_weights = {0: 0.8, 1: 0.5, 2: 0.2}
    string_class_weights = {"bird": 0.8, "dog": 0.5, "cat": 0.2}
    
    return conf_matrix, cost_matrix, class_weights

def test_macro_average(setup_data):
    conf_matrix, cost_matrix, class_weights = setup_data
    cost = utils.calculate_weighted_cost(conf_matrix, cost_matrix, method='macro')
    assert cost == pytest.approx(-0.67, rel=1e-2)

def test_weighted_average_with_class_weights(setup_data):
    conf_matrix, cost_matrix, class_weights = setup_data
    cost = utils.calculate_weighted_cost(conf_matrix, cost_matrix, method='weighted', class_weights=class_weights)
    assert cost == pytest.approx(-4.3, rel=1e-2)

def test_weighted_average_without_class_weights(setup_data):
    conf_matrix, cost_matrix, class_weights = setup_data
    cost = utils.calculate_weighted_cost(conf_matrix, cost_matrix, method='weighted')
    assert cost == pytest.approx(-1.8, rel=1e-2)

def test_micro_average(setup_data):
    conf_matrix, cost_matrix, class_weights = setup_data
    cost = utils.calculate_weighted_cost(conf_matrix, cost_matrix, method='micro')
    assert cost == pytest.approx(-2.0, rel=1e-2)

def test_invalid_method(setup_data):
    conf_matrix, cost_matrix, class_weights = setup_data
    with pytest.raises(ValueError, match="Unsupported method: invalid_method"):
        utils.calculate_weighted_cost(conf_matrix, cost_matrix, method='invalid_method')

def test_mismatched_class_weights(setup_data):
    conf_matrix, cost_matrix, class_weights = setup_data
    invalid_class_weights = {0: 0.8, 1: 0.5}  # Only two classes provided
    with pytest.raises(AssertionError, match="Mismatch between class weights and confusion matrix classes."):
        utils.calculate_weighted_cost(conf_matrix, cost_matrix, method='weighted', class_weights=invalid_class_weights)


# Run the tests
if __name__ == "__main__":
    pytest.main()