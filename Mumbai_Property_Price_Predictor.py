# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

sns.set(style="whitegrid")

# Load dataset
df = pd.read_csv("Mumbai1.csv")

# Drop unnecessary column
df = df.drop(columns=['Unnamed: 0'])

# Rename columns
df.rename(columns={
    'Price': 'price',
    'Area': 'area',
    'Location': 'location',
    'No. of Bedrooms': 'bedrooms',
    'New/Resale': 'new_resale'
}, inplace=True)

# Remove missing values
df = df.dropna()

# Convert columns to numeric
df['price'] = pd.to_numeric(df['price'])
df['area'] = pd.to_numeric(df['area'])
df['bedrooms'] = pd.to_numeric(df['bedrooms'])

# Create price per sqft feature
df['price_per_sqft'] = df['price'] / df['area']

# Sample data for cleaner scatter plot
df_sample = df.sample(n=500, random_state=42)

# Plot area vs price
plt.figure(figsize=(7,5))
sns.scatterplot(data=df_sample, x='area', y='price', alpha=0.4, color='blue')
plt.title("Area vs Price (Sampled Data)")
plt.show()
print("Insight: Property prices generally increase with area, but there is high variation due to other factors like location.")

# Plot bedrooms vs price
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x='bedrooms', y='price', palette="Greens")
plt.title("Bedrooms vs Price")
plt.show()
print("Insight: Properties with more bedrooms tend to have higher prices, though variability exists within each category.")

# Top 10 locations by average price
avg_price = df.groupby('location')['price'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
avg_price.plot(kind='bar', color='orange')
plt.title("Top 10 Locations by Average Price")
plt.xticks(rotation=45)
plt.show()
print("Insight: Certain locations clearly command premium pricing, indicating higher demand or better infrastructure.")

# Hexbin plot for density
plt.figure(figsize=(7,5))
plt.hexbin(df['area'], df['price'], gridsize=30, cmap='Blues')
plt.colorbar(label='Number of Properties')
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Density Plot (Area vs Price)")
plt.show()
print("Insight: Most properties are concentrated in mid-range area and price segments, showing market clustering.")

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
print("Insight: Area has a strong positive correlation with price, making it a key predictor.")

# Encode categorical variables
df = pd.get_dummies(df, columns=['location'], drop_first=True)

# Define features and target
X = df.drop(columns=['price'])
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Evaluate model
print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot actual vs predicted line graph
plt.figure(figsize=(8,5))
plt.plot(y_test.values, label='Actual', color='black')
plt.plot(y_pred, label='Predicted', color='red')
plt.legend()
plt.title("Actual vs Predicted Prices")
plt.show()
print("Insight: The model predictions generally follow actual prices, but deviations exist indicating room for improvement.")

# Plot scatter graph
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color='green')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Scatter")
plt.show()
print("Insight: Points closer to the diagonal indicate better predictions, while spread shows prediction error.")

# Show feature importance
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("\nFeature Importance:")
print(coefficients.sort_values(by='Coefficient', ascending=False))
print("Insight: Features with higher coefficients have stronger influence on property price.")

# Save cleaned dataset
df.to_csv("clean_mumbai_real_estate.csv", index=False)
print("\nClean dataset saved successfully.")