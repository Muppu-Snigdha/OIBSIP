import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("data.csv")

# =========================================================
# PROJECT TITLE
# =========================================================

print("\n===================================================")
print(" AIRBNB EXPLORATORY DATA ANALYSIS PROJECT ")
print("===================================================\n")

# =========================================================
# DATASET OVERVIEW
# =========================================================

print("Dataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# =========================================================
# DATA CLEANING
# =========================================================

print("\n===================================================")
print(" DATA CLEANING ")
print("===================================================\n")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Remove missing values
df.dropna(inplace=True)

print("Duplicates Removed Successfully!")

print("Missing Values Removed Successfully!")

# =========================================================
# STANDARDIZATION
# =========================================================

print("\n===================================================")
print(" STANDARDIZATION ")
print("===================================================\n")

scaler = StandardScaler()

df["scaled_price"] = scaler.fit_transform(
    df[["price"]]
)

print(df[["price", "scaled_price"]].head())

# =========================================================
# GRAPH STYLE
# =========================================================

sns.set_style("darkgrid")

# =========================================================
# GRAPH 1
# NEIGHBOURHOOD GROUP DISTRIBUTION
# =========================================================

plt.figure(figsize=(10,5))

sns.countplot(

    x="neighbourhood_group",

    data=df,
    hue = "neighbourhood_group",
    palette="Set2",
    legend=False
)

plt.title(
    "Neighbourhood Group Distribution",
    fontsize=16
)

plt.xlabel("Neighbourhood Group")

plt.ylabel("Count")

plt.tight_layout()

plt.show()

# =========================================================
# GRAPH 2
# PRICE DISTRIBUTION
# =========================================================

plt.figure(figsize=(10,5))

sns.histplot(

    df["price"],

    bins=30,

    kde=True,

    color="orange"
)

plt.title(
    "Price Distribution",
    fontsize=16
)

plt.xlabel("Price")

plt.ylabel("Frequency")

plt.tight_layout()

plt.show()

# =========================================================
# GRAPH 3
# ROOM TYPE DISTRIBUTION
# =========================================================

room_counts = df["room_type"].value_counts()

plt.figure(figsize=(7,7))

plt.pie(

    room_counts,

    labels=room_counts.index,

    autopct="%1.1f%%",

    startangle=90
)

plt.title(
    "Room Type Distribution",
    fontsize=16
)

plt.tight_layout()

plt.show()

# =========================================================
# GRAPH 4
# REVIEWS VS PRICE
# =========================================================

plt.figure(figsize=(10,5))

sns.scatterplot(

    x="number_of_reviews",

    y="price",

    data=df,

    hue="room_type"
)

plt.title(
    "Reviews vs Price",
    fontsize=16
)

plt.xlabel("Number of Reviews")

plt.ylabel("Price")

plt.tight_layout()

plt.show()

# =========================================================
# GRAPH 5
# AVAILABILITY ANALYSIS
# =========================================================

plt.figure(figsize=(10,5))

sns.boxplot(

    x="room_type",

    y="availability_365",

    data=df,
    hue = "room_type",

    palette="Set3",
    legend=False
)

plt.title(
    "Availability by Room Type",
    fontsize=16
)

plt.xlabel("Room Type")

plt.ylabel("Availability (365 Days)")

plt.tight_layout()

plt.show()

# =========================================================
# GRAPH 6
# OUTLIER DETECTION
# =========================================================

plt.figure(figsize=(10,5))

sns.boxplot(

    x=df["price"],

    color="red"
)

plt.title(
    "Outlier Detection in Price",
    fontsize=16
)

plt.xlabel("Price")

plt.tight_layout()

plt.show()

# =========================================================
# GRAPH 7
# TOP 10 EXPENSIVE NEIGHBOURHOODS
# =========================================================

top_places = df.groupby(
    "neighbourhood_group"
)["price"].mean().sort_values(ascending=False)

plt.figure(figsize=(10,5))

top_places.plot(

    kind="bar",

    color="purple"
)

plt.title(
    "Average Price by Neighbourhood Group",
    fontsize=16
)

plt.xlabel("Neighbourhood Group")

plt.ylabel("Average Price")

plt.tight_layout()

plt.show()

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n===================================================")
print(" PROJECT SUMMARY ")
print("===================================================\n")

print("1. Data Cleaning Completed")

print("2. Missing Values Removed")

print("3. Duplicate Records Removed")

print("4. Standardization Applied")

print("5. Outlier Detection Performed")

print("6. Graphical Analysis Completed")

print("\nEDA Project Completed Successfully!")