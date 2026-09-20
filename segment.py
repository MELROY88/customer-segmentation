"""Customer segmentation with K-Means on sample (synthetic) data."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(7)
df = pd.DataFrame({
    "recency_days": np.r_[rng.integers(1, 30, 200), rng.integers(30, 120, 250), rng.integers(120, 365, 150)],
    "orders": np.r_[rng.integers(8, 25, 200), rng.integers(2, 8, 250), rng.integers(1, 3, 150)],
    "avg_spend": np.r_[rng.normal(180, 30, 200), rng.normal(90, 20, 250), rng.normal(50, 15, 150)],
})
X = StandardScaler().fit_transform(df)
df["segment"] = KMeans(n_clusters=3, n_init=10, random_state=1).fit_predict(X)
summary = df.groupby("segment").mean().round(1)
summary["customers"] = df.segment.value_counts().sort_index()
print(summary)
summary.to_csv("segment_summary.csv")

plt.figure(figsize=(6, 4))
plt.scatter(df.orders, df.avg_spend, c=df.segment, cmap="viridis", s=12)
plt.xlabel("Orders"); plt.ylabel("Average spend"); plt.title("Customer segments (sample data)")
plt.tight_layout(); plt.savefig("segments.png", dpi=150)
