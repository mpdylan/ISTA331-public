''' Don't post, this is in the solution pdf '''
import matplotlib.pyplot as plt
import statsmodels.api as sm, numpy as np

def get_parabola(xy_lst):
    x = np.array([coords[0] for coords in xy_lst])
    y_lst = [coords[1] for coords in xy_lst]
    X = np.column_stack([x, x**2])
    X = sm.add_constant(X)
    model = sm.OLS(y_lst, X)
    results = model.fit()
    # results.params is an array because we fit to a list, not a Series
    # order is reversed (sort of) compared to if it was a Series
    return results.params[2], results.params[1], results.params[0]
    
xy_lst = [[2, 1.07], [1.3, 0], [2.8, 0], [0.7, -2], [3.4, -2], [0.5, -3],
          [3, -1], [0.4, -4]]
a, b, c = get_parabola(xy_lst)

x = np.array([coords[0] for coords in xy_lst])  
y_lst = [coords[1] for coords in xy_lst]  
plt.plot(x, y_lst, linestyle='', marker='o')

x = np.linspace(0, 4)
y = a * x**2 + b * x + c
plt.plot(x, y, linestyle='--')

plt.grid()
plt.show()
    