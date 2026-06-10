import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# ---------------------------
# Load Dataset
# ---------------------------

df = pd.read_csv("data/student_performance.csv")

print("="*50)
print("DATASET INFORMATION")
print("="*50)

print(df.head())
print("\nDataset Shape:", df.shape)

print("\nData Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Records")
print(df.duplicated().sum())

# ---------------------------
# Data Cleaning
# ---------------------------

df.drop_duplicates(inplace=True)

for col in df.select_dtypes(include=np.number):
    df[col].fillna(df[col].mean(), inplace=True)

# ---------------------------
# Create Average Score
# ---------------------------

score_columns = [
    "Math_Score",
    "Science_Score",
    "English_Score"
]

df["Average_Score"] = df[score_columns].mean(axis=1)

# ---------------------------
# Descriptive Statistics
# ---------------------------

print("\nDescriptive Statistics")
print(df.describe())

# ---------------------------
# Visualization Settings
# ---------------------------

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8,5)

# ---------------------------
# 1. Study Hours vs Score
# ---------------------------

plt.figure()
sns.scatterplot(
    x="Study_Hours",
    y="Average_Score",
    data=df
)

plt.title("Study Hours vs Average Score")
plt.savefig(
    "images/study_hours_vs_score.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ---------------------------
# 2. Attendance vs Score
# ---------------------------

plt.figure()
sns.scatterplot(
    x="Attendance",
    y="Average_Score",
    data=df
)

plt.title("Attendance vs Average Score")
plt.savefig(
    "images/attendance_vs_score.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ---------------------------
# 3. Subject Average Scores
# ---------------------------

subject_means = df[
    ["Math_Score",
     "Science_Score",
     "English_Score"]
].mean()

plt.figure()

subject_means.plot(kind="bar")

plt.title("Subject Wise Average Scores")
plt.ylabel("Average Score")

plt.savefig(
    "images/subject_average_scores.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ---------------------------
# 4. Correlation Heatmap
# ---------------------------

plt.figure(figsize=(10,6))

corr_matrix = df.corr(numeric_only=True)

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig(
    "images/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ---------------------------
# 5. Score Distribution
# ---------------------------

plt.figure()

sns.histplot(
    df["Average_Score"],
    kde=True,
    bins=20
)

plt.title("Distribution of Average Scores")

plt.savefig(
    "images/score_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ---------------------------
# Hypothesis Testing
# ---------------------------

corr, p_value = pearsonr(
    df["Study_Hours"],
    df["Average_Score"]
)

print("\nCorrelation Test")

print("Correlation:", round(corr,3))
print("P-value:", p_value)

if p_value < 0.05:
    print("Significant relationship found.")
else:
    print("No significant relationship found.")

print("\nEDA Completed Successfully.")