#imports
import numpy as np


def wolfe_condition_with_backtracking(f, x, curr_val, curr_gradient, direction, alpha=0.01, beta=0.5):
    """
    Perform backtracking line search using the Wolfe condition.
    alpha: Initial step size.
    beta: Reduction factor for the step size - backtracking constant.
    """
    step_size = alpha
    while True:
        next_x = x + step_size * direction
        next_val, _, _ = f(next_x, False)
        # Check sufficient decrease in objective function
        if next_val <= curr_val + alpha * step_size * np.dot(curr_gradient, direction):
            break
        step_size *= beta
    return step_size

class LineSearchMinimizationWithGradientDescent:
    """
        f: The function to be minimized.
        x0: The starting point for the minimization.
        obj_tol: The numeric tolerance for successful termination in terms of a small enough
                change in objective function values between two consecutive iterations 
                (f(x_{i+1}) and f(x_i)).
        param_tol: The numeric tolerance for successful termination in terms of a small enough
                  distance between two consecutive iteration locations (x_{i+1} and x_i).
        max_iter: The maximum allowed number of iterations.
        iterations: enables access to the entire path of iterations when done
        objective_values: enable access to the objective values when done
    """
    def __init__(self, f, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100):
        self.f = f
        self.x0 = x0
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter
        self.iterations = []
        self.objective_values = []
    
    def get_path(self):
        return self.iterations, self.objective_values

    def minimize(self):
        x = self.x0
        curr_val, curr_gradient, _ = self.f(x, True)
        
        for i in range(self.max_iter):
            # Report current iteration details
            print(f"Iteration {i}: x_i = {x}, f(x) = {curr_val}")

            # Save current state
            self.iterations.append(x.copy())
            self.objective_values.append(curr_val)

            # Direction is the negative gradient
            direction = -curr_gradient

            # Perform backtracking line search using the Wolfe condition - will help us find the optimal step size
            step_size = wolfe_condition_with_backtracking(self.f, x, curr_val, curr_gradient, -curr_gradient)

            # Update the current position
            next_x = x + step_size * direction
            next_val, next_gradient, _ = self.f(next_x, True)

            # Check for convergence
            if np.abs(next_val - curr_val) < self.obj_tol or np.linalg.norm(next_x - x) < self.param_tol:
                print(f"Convergence achieved after {i+1} iterations.")
                self.iterations.append(next_x.copy())
                self.objective_values.append(next_val)
                return next_x, next_val, True
            
            # Update variables for the next iteration
            x, curr_val, curr_gradient = next_x, next_val, next_gradient
    
        print("Maximum iterations reached without convergence.")
        return x, curr_val, False
    


class LineSearchMinimizationWithNewton:
    """
        f: The function to be minimized.
        x0: The starting point for the minimization.
        obj_tol: The numeric tolerance for successful termination in terms of a small enough
                change in objective function values between two consecutive iterations 
                (f(x_{i+1}) and f(x_i)).
        param_tol: The numeric tolerance for successful termination in terms of a small enough
                  distance between two consecutive iteration locations (x_{i+1} and x_i).
        max_iter: The maximum allowed number of iterations.
        iterations: enables access to the entire path of iterations when done
        objective_values: enable access to the objective values when done
    """
    def __init__(self, f, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100):
        self.f = f
        self.x0 = x0
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter
        self.iterations = []
        self.objective_values = []
    
    def get_path(self):
        return self.iterations, self.objective_values
    

    def minimize(self):
        x = self.x0
        curr_val, curr_gradient, curr_hessian = self.f(x, True)
        
        for i in range(self.max_iter):
            # Report current iteration details
            print(f"Iteration {i}: x_i = {x}, f(x) = {curr_val}")
            
            # Save current state
            self.iterations.append(x.copy())
            self.objective_values.append(curr_val)
            
            # Newton direction
            direction = -np.linalg.solve(curr_hessian, curr_gradient)
            
            # Perform backtracking line search using the Wolfe condition - will help us find the optimal step size
            step_size = wolfe_condition_with_backtracking(self.f, x, curr_val, curr_gradient, direction)
            
            # Update the current position
            next_x = x + step_size * direction
            next_val, next_gradient, next_hessian = self.f(next_x, True)
            
            # Check for convergence
            if np.abs(next_val - curr_val) < self.obj_tol or np.linalg.norm(next_x - x) < self.param_tol:
                print(f"Convergence achieved after {i+1} iterations.")
                self.iterations.append(next_x.copy())
                self.objective_values.append(next_val)
                return next_x, next_val, True
            
            # Update variables for the next iteration
            x, curr_val, curr_gradient, curr_hessian = next_x, next_val, next_gradient, next_hessian
        
        print("Maximum iterations reached without convergence.")
        return x, curr_val, False