import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import datasets
from sklearn import manifold


data = datasets.fetch_openml("mnist_784", version=1, return_X_y=True)

pixel_values, targets = data
targets = targets.astype(int)

single_image = pixel_values[:1].values.reshape(28, 28)
plt.imshow(single_image)
plt.show()


tsne = manifold.TSNE(n_components=2, random_state=42)
transformed_data = tsne.fit_transform(pixel_values[:3000].values)

# converted to a pandas dataframe
tsne_df = pd.DataFrame(np.column_stack((transformed_data, targets[:3000].values)), columns=["x", "y", "targets"])
tsne_df.loc[:, "targets"] = tsne_df.targets.astype(int)

# Plot using seaborn and matplotlib
grid = sns.FacetGrid(tsne_df, hue="targets", size=8)
grid.map(plt.scatter, "x", "y").add_legend()
plt.show()