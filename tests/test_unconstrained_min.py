import unittest
import numpy as np
from src.unconstrained_min import LineSearchMinimizationWithGradientDescent, LineSearchMinimizationWithNewton
from src.utils import plot_contours, plot_function_values
from tests.examples import quadratic_function_1, quadratic_function_2, quadratic_function_3, rosenbrock_function, linear_function, exponential_function

class TestUnconstrainedMinimization(unittest.TestCase):
    def setUp(self):
        self.initial_points = {
            "default": np.array([1.0, 1.0]),
            "rosenbrock": np.array([-1.0, 2.0])
        }
        self.max_iterations = {
            "default": 100,
            "rosenbrock_gd": 10000
        }
        self.functions = [
            ("Quadratic Function 1", quadratic_function_1),
            ("Quadratic Function 2", quadratic_function_2),
            ("Quadratic Function 3", quadratic_function_3),
            ("Rosenbrock Function", rosenbrock_function),
            ("Linear Function", linear_function),
            ("Exponential Function", exponential_function)
        ]
    
    def run_minimization(self, func, x0, func_name):
        
        # Instantiate minimizers
        if(func_name == "Rosenbrock Function"):
            gd_minimizer = LineSearchMinimizationWithGradientDescent(func, x0, max_iter=1000)
        else:
            gd_minimizer = LineSearchMinimizationWithGradientDescent(func, x0, max_iter=1000)

        newton_minimizer = LineSearchMinimizationWithNewton(func, x0, max_iter=100)

        # Run minimization
        gd_result = gd_minimizer.minimize()
        newton_result = newton_minimizer.minimize()

        # Get paths and objective values
        gd_path, gd_obj_values = gd_minimizer.get_path()
        newton_path, newton_obj_values = newton_minimizer.get_path()

        return (gd_result, gd_path, gd_obj_values), (newton_result, newton_path, newton_obj_values)
    
    def test_functions(self):
        for func_name, func in self.functions:
            if func_name == "Rosenbrock Function":
                x0 = self.initial_points["rosenbrock"]
            else:
                x0 = self.initial_points["default"]

            (gd_result, gd_path, gd_obj_values), (newton_result, newton_path, newton_obj_values) = self.run_minimization(func, x0, func)
            print(f"Gradient Descent - {func_name}: {gd_result}")
            print(f"Newton's Method - {func_name}: {newton_result}")

            self.plot_results(func_name, func, gd_path, newton_path, gd_obj_values, newton_obj_values)

    def plot_results(self, func_name, func, gd_path, newton_path, gd_obj_values, newton_obj_values):
        # Plot contour lines with iteration paths
        plot_contours(
            func,
            x_limits=(-3, 3),
            y_limits=(-3, 3),
            title=f'Contour Plot - {func_name}',
            paths=[gd_path, newton_path],
            path_names=["Gradient Descent", "Newton's Method"]
        )

        # Plot function values vs. iteration number
        plot_function_values(
            [gd_obj_values, newton_obj_values],
            method_names=["Gradient Descent", "Newton's Method"],
            title=f'Function Values vs. Iteration - {func_name}',
        )

if __name__ == '__main__':
    unittest.main()