import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv(r'balance-scale.csv')
    df.head()

    x_axis_labels = ["Class", "L-Weight", "L-Distance", "R-Weight", "R-Distance"] # labels for x-axis

    num_feat = df.select_dtypes(include=np.number).columns
    plt.figure(figsize=(15, 15))
    sns.heatmap(df.corr(), cmap="Blues", annot=True,xticklabels= x_axis_labels, yticklabels=x_axis_labels)
    plt.show()