''' Not for posting, this is how I made the plots for the worksheet. '''
import matplotlib.pyplot as plt
import numpy as np

# fig 1
X = np.linspace(-2, 3)
y = [x**2 for x in X]
plt.plot(X, y)

y = [-2 * (x - 2)**2 + 1 for x in X]
plt.plot(X, y, linestyle='--')

ax = plt.gca()
ax.set_ylim(-4, 4)

plt.grid()

ax.text(-1, 2, r'$y = x^2$', fontsize=24)
ax.text(1.25, -2, r'$y = ?$', fontsize=24)

# fig 2
X = np.linspace(-np.pi, 3 * np.pi, 1000)
y = [np.sin(x) for x in X]
plt.plot(X, y)

y = [2 * np.sin(x / 2 - np.pi / 2) + 3 for x in X]
plt.plot(X, y, linestyle='--')

plt.grid()

ax = plt.gca()
ax.text(5.5, 1.5, r'$y = sin(x)$', fontsize=24)
ax.text(-3, 3.4, r'$y = ?$', fontsize=24)




