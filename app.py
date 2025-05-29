import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

attribute_names = [
    "buying", "maint", "doors", "persons", "lug_boot", "safety", "class"
]

df = pd.read_csv(
    "carEvol/car.data",
    header=None,
    names=attribute_names
)

features = attribute_names[:-1]
target = attribute_names[-1]

X = pd.get_dummies(df[features])
y = df[target]

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_stdized = (X - X_mean) / X_std

cov_matrix = np.cov(X_stdized, rowvar=False)
eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
sorted_idx = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[sorted_idx]
eig_vecs = eig_vecs[:, sorted_idx]

PCs = X_stdized.values @ eig_vecs[:, :2]
explained_var = eig_vals[:2] / eig_vals.sum()

pca_df = pd.DataFrame(PCs, columns=['PC1', 'PC2'])
pca_df['class'] = y

plt.figure(figsize=(5, 4))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='class', s=60, palette='deep')
plt.title("Car Evaluation PCA Projection (2D)")
plt.xlabel(f"Principal Component 1 ({explained_var[0]*100:.2f}% variance)")
plt.ylabel(f"Principal Component 2 ({explained_var[1]*100:.2f}% variance)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='class', s=60, palette='deep')
for i, feature in enumerate(X.columns[:5]):
    plt.arrow(0, 0, eig_vecs[i, 0]*2.5, eig_vecs[i, 1]*2.5, color='red', alpha=0.5, head_width=0.08)
    plt.text(eig_vecs[i, 0]*2.8, eig_vecs[i, 1]*2.8, feature, color='black', fontsize=7)
plt.title("Car Evaluation PCA Biplot (partial features)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='class', s=60, palette='deep')

dummy_features = [col for i, col in enumerate(X.columns) if i == 0 or X.columns[i].split('_')[0] != X.columns[i-1].split('_')[0]]

for feature in dummy_features:
    idx = X.columns.get_loc(feature)
    plt.arrow(0, 0, eig_vecs[idx, 0]*5, eig_vecs[idx, 1]*5, color='red', alpha=0.5, head_width=0.12)
    plt.text(eig_vecs[idx, 0]*5.5, eig_vecs[idx, 1]*5.5, feature, color='black', fontsize=7, ha='left', va='bottom')

plt.title("Car Evaluation PCA Biplot (less overlap)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()