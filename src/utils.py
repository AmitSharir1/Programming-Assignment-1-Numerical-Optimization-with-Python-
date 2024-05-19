import numpy as np
import matplotlib.pyplot as plt

def plot_contours(f, x_limits, y_limits, title, levels=50, paths=None, path_names=None):
    """
    Plots contour lines of the function f within the given limits.
    
    Parameters:
    - f: The objective function, which returns the value at a given point.
    - x_limits: A tuple (x_min, x_max) specifying the limits for the x-axis.
    - y_limits: A tuple (y_min, y_max) specifying the limits for the y-axis.
    - levels: The number of contour levels to plot.
    - title: The title of the plot.
    - paths: A list of paths (each path is a list of points) to plot.
    - path_names: A list of names for the paths, to be used in the legend.
    """
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    
    x = np.linspace(x_min, x_max, 400)
    y = np.linspace(y_min, y_max, 400)
    X, Y = np.meshgrid(x, y)
    
    Z = np.array([[f(np.array([xx, yy]), False)[0] for xx, yy in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])
    
    plt.figure(figsize=(10, 8))
    cp = plt.contour(X, Y, Z, levels=levels)
    plt.colorbar(cp)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    if paths is not None:
        for i, path in enumerate(paths):
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], marker='o', label=path_names[i] if path_names else f'Path {i+1}')
        plt.legend()
    
    plt.show()

def plot_function_values(iteration_values, method_names, title):
    """
    Plots function values at each iteration for given methods on the same plot to enable comparison.
    
    Parameters:
    - iteration_values: A list of lists, where each sublist contains function values at each iteration for a method.
    - method_names: A list of names for the methods.
    - title: The title of the plot.
    """
    plt.figure(figsize=(10, 8))
    
    for values, name in zip(iteration_values, method_names):
        plt.plot(values, marker='o', label=name)
    
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.legend()
    plt.grid(True)
    plt.show()