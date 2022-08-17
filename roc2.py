import matplotlib.pyplot as plt
plt.plot(29, 450, linestyle='--', label='Random prediction (AUROC = 0.888)')
plt.plot(45, 446, marker='.', label='Random Forest (AUROC = 0.880)')
plt.plot(42, 429, marker='.', label='Naive Bayes (AUROC = 0.786)')

# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() #
# Show plot
plt.show()