import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from statistics import mean
from global_utils import load_results

results = load_results('exp3_1_results.p')
ami_samples = results['ami_samples']

fig = plt.figure()
ax = fig.gca(projection='3d')

x_lin = results['label_counts']
y_lin = results['constraint_counts']
X_arr, Y_arr = np.meshgrid(x_lin, y_lin, indexing='ij')
Z_arr = np.zeros(X_arr.shape)

for i in range(len(x_lin)):
    for j in range(len(y_lin)):
        x = X_arr[i, j]
        y = Y_arr[i, j]
        Z_arr[i, j] = mean(ami_samples[x][y])

ax.plot_surface(X_arr, Y_arr, Z_arr, cmap=cm.coolwarm)
ax.set_xlabel('Label Counts')
ax.set_ylabel('Constraint Counts')
ax.set_zlabel('AMI')

plt.title('Performance Evaluation of Labels and Constraints')
plt.show()
