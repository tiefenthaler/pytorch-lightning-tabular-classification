import unittest
import numpy as np
from sklearn.metrics import confusion_matrix

# import custom functions
import sys
sys.path.append('/Users/dat/Library/CloudStorage/OneDrive-foryouandyourcustomers/GitHub/AutomatedPackagingCategories_Showcase/ml_packaging_classification/src')
import utils


class TestCalculateWeightedCost(unittest.TestCase):

    def setUp(self):
        # Using sklearn confusion_matrix
        y_true = ["cat", "dog", "bird", "cat", "bird"]
        y_pred = ["cat", "dog", "cat", "cat", "bird"]
        class_labels = np.sort(["bird", "dog", "cat"])  # the order in which classes appear in the confusion matrix
        self.conf_matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
        
        self.cost_matrix = [[1, -10], [-5, 1]]
        self.class_weights = {0: 0.8, 1: 0.5, 2: 0.2}
        self.string_class_weights = {"bird": 0.8, "dog": 0.5, "cat": 0.2}

    def test_macro_average(self):
        cost = utils.calculate_weighted_cost(self.conf_matrix, self.cost_matrix, method='macro')
        self.assertAlmostEqual(cost, -3.33, places=2)

    def test_weighted_average_with_strings(self):
        cost = utils.calculate_weighted_cost(self.conf_matrix, self.cost_matrix, method='weighted', class_weights=self.string_class_weights)
        self.assertAlmostEqual(cost, -1.67, places=2)

    def test_weighted_average(self):
        cost = utils.calculate_weighted_cost(self.conf_matrix, self.cost_matrix, method='weighted', class_weights=self.class_weights)
        self.assertAlmostEqual(cost, -1.67, places=2)

    def test_micro_average(self):
        cost = utils.calculate_weighted_cost(self.conf_matrix, self.cost_matrix, method='micro')
        self.assertAlmostEqual(cost, -10.0, places=2)

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            utils.calculate_weighted_cost(self.conf_matrix, self.cost_matrix, method='invalid_method')

    def test_mismatched_class_weights(self):
        invalid_class_weights = {0: 0.8, 1: 0.5}
        with self.assertRaises(AssertionError):
            utils.calculate_weighted_cost(self.conf_matrix, self.cost_matrix, method='weighted', class_weights=invalid_class_weights)

    def test_weighted_average_without_class_weights(self):
        cost = utils.calculate_weighted_cost(self.conf_matrix, self.cost_matrix, method='weighted')
        self.assertAlmostEqual(cost, -3.33, places=2)

if __name__ == "__main__":
    unittest.main()
