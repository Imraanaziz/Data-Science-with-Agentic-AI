import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# --- Replace these with your actual results from the model outputs ---
models = ['Logistic Regression', 'KNN', 'SVM']

# From your output
accuracy = [0.8163, 0.7928, 0.8242]
precision = [0.74, 0.53, 0.69]
recall = [0.23, 0.33, 0.34]
f1_score = [0.35, 0.41, 0.45]

# Convert to numpy arrays
metrics = np.array([accuracy, precision, recall, f1_score])

# Set up 3D bar chart
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

xpos = np.arange(len(models))
ypos = np.arange(metrics.shape[0])
xpos, ypos = np.meshgrid(xpos, ypos, indexing="ij")

xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)

# Flatten metric values for bar height
dz = metrics.T.flatten()

# Bar size
dx = dy = 0.4

# Labels for metrics
metric_labels = ['Accuracy', 'Precision (C1)', 'Recall (C1)', 'F1-score (C1)']

# Colors based on metric type
colors = ['skyblue', 'lightgreen', 'salmon', 'plum']
bar_colors = [colors[i % 4] for i in ypos]

# Plot bars
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=bar_colors, shade=True)

# Set axis labels
ax.set_xlabel('Model')
ax.set_ylabel('Metric')
ax.set_zlabel('Score')
ax.set_title('3D Comparison of Model Performance Metrics')

# Custom tick labels
ax.set_xticks(np.arange(len(models)) + dx / 2)
ax.set_xticklabels(models, rotation=15)

ax.set_yticks(np.arange(len(metric_labels)) + dy / 2)
ax.set_yticklabels(metric_labels)

plt.tight_layout()
plt.show()

