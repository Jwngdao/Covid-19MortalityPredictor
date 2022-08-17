import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

plt.plot(29, 450, linestyle='--', label='Decision Tree' )
plt.plot(45, 446, marker='.', label='Support Vector Machine' )
plt.plot(42, 429, marker='.', label='k Nearest Neighbor' )

# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() #
# Show plot
plt.show()