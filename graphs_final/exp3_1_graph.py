import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from global_utils import load_results

results = load_results('exp3_1_results.p')
ami_samples = results['ami_samples']

fig = plt.figure()
ax = fig.gca(projection='3d')

x_lin = results['label_counts']
y_lin = results['constraint_counts']
X_arr, Y_arr = np.meshgrid(x_lin, y_lin)

Z = []
for i, y in enumerate(y_lin):
    Z.append([])
    for j, x in enumerate(x_lin):
        average_ami = mean(ami_samples[x][y])
        Z[i].append(average_ami)

Z_arr = np.array(Z)

ax.plot_surface(X_arr, Y_arr, Z_arr)
ax.set_xlabel('Label Counts')
ax.set_ylabel('Constraints Counts')
ax.set_zlabel('AMI')

plt.title('Performance Evaluation of Labels and Constraints')
plt.show()
