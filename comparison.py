import matplotlib.pyplot as plt
import numpy as np

# 1. Model Names (All 6 Models)
models = ['Logistic \nRegression', 'Random \nForest', 'Decision \nTree', 'SVM', 'KNN', 'Gradient \nBoosting']

# 2. Performance Metrics (Percentages out of 100, Class 0 & 1 Averaged)
accuracy = [94.9, 89.9, 88.6, 88.6, 87.3, 88.6]
precision = [94.5, 88.5, 87.0, 87.5, 86.5, 87.5] 
recall = [94.5, 89.5, 87.5, 86.5, 85.0, 86.5]
f1_score = [94.5, 89.0, 87.5, 87.0, 85.5, 87.0]

# 3. Graph Settings
x = np.arange(len(models))  # X-axis positions
width = 0.2  # Width of each individual bar

fig, ax = plt.subplots(figsize=(14, 6))

# 4. Plotting the 4 bars for each model side-by-side
rects1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#4C1D95')  # Purple
rects2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='#06B6D4') # Cyan
rects3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='#10B981')    # Green
rects4 = ax.bar(x + 1.5*width, f1_score, width, label='F1-Score', color='#F59E0B')  # Orange

# 5. Formatting the Graph to look Professional
ax.set_ylabel('Scores (%)', fontsize=12, fontweight='bold')
ax.set_title('Overall Performance Comparison of Machine Learning Models', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')

# Setting Y-axis from 75 to 100 so the differences are clearly visible
ax.set_ylim(75, 100) 
ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)

# Add grid lines behind the bars for easy reading
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add exact numbers on top of the bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, rotation=90)

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)
add_labels(rects4)

# 6. Save and Show
plt.tight_layout()
plt.savefig("model_comparison_bar_chart_final.png", dpi=300) # Saves a high-quality image
plt.show()