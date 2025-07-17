#!/usr/bin/env python3
"""
🚌 Public Transport Cost Optimization - Complete ML Project
A comprehensive data analytics and machine learning project to predict and optimize travel costs.
"""

# ✅ STEP 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("📚 All libraries imported successfully!")

# ✅ STEP 2: Load or Generate Dataset


# ✅ STEP 2: Exploratory Data Analysis (EDA)
print("\n" + "="*50)
print("📊 EXPLORATORY DATA ANALYSIS")
print("="*50)

# Load dataset
df = pd.read_csv('transport_data.csv')

# Basic info
print("\n📋 Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print("\n🔍 First 5 rows:")
print(df.head())

print("\n📈 Dataset Info:")
print(df.info())

print("\n📊 Statistical Summary:")
print(df.describe())

# Check for missing values and duplicates
print(f"\n❓ Missing values: {df.isnull().sum().sum()}")
print(f"🔄 Duplicate rows: {df.duplicated().sum()}")

# Data visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('📊 Public Transport Cost Analysis Dashboard', fontsize=16, fontweight='bold')

# 1. Scatterplot: Distance vs Total Cost
axes[0, 0].scatter(df['distance_km'], df['total_cost'], alpha=0.6, color='#1f77b4')
axes[0, 0].set_xlabel('Distance (km)')
axes[0, 0].set_ylabel('Total Cost ($)')
axes[0, 0].set_title('Distance vs Total Cost')
axes[0, 0].grid(True, alpha=0.3)

# 2. Bar chart: Cost per Time Slot
cost_by_time = df.groupby('time_slot')['total_cost'].mean()
colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
bars = axes[0, 1].bar(cost_by_time.index, cost_by_time.values, color=colors)
axes[0, 1].set_xlabel('Time Slot')
axes[0, 1].set_ylabel('Average Total Cost ($)')
axes[0, 1].set_title('Average Cost by Time Slot')
axes[0, 1].tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:.1f}', ha='center', va='bottom')

# 3. Correlation heatmap
numeric_cols = ['distance_km', 'fuel_cost', 'maintenance_cost', 'no_passengers', 'travel_duration_min', 'total_cost']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
axes[1, 0].set_title('Feature Correlation Matrix')

# 4. Boxplot: Passengers vs Cost
df['passenger_group'] = pd.cut(df['no_passengers'], bins=[0, 10, 20, 30, 50], labels=['Low', 'Medium', 'High', 'Very High'])
sns.boxplot(data=df, x='passenger_group', y='total_cost', ax=axes[1, 1])
axes[1, 1].set_xlabel('Passenger Groups')
axes[1, 1].set_ylabel('Total Cost ($)')
axes[1, 1].set_title('Cost Distribution by Passenger Groups')

plt.tight_layout()
plt.show()

# ✅ STEP 3: Data Preprocessing
print("\n" + "="*50)
print("🔧 DATA PREPROCESSING")
print("="*50)

# Create a copy for preprocessing
df_processed = df.copy()

# Encode categorical features
print("\n🏷️ Encoding categorical variables...")
le_route = LabelEncoder()
le_time = LabelEncoder()

df_processed['route_id_encoded'] = le_route.fit_transform(df_processed['route_id'])
df_processed['time_slot_encoded'] = le_time.fit_transform(df_processed['time_slot'])

# Alternative: One-hot encoding for time_slot (more interpretable)
time_dummies = pd.get_dummies(df_processed['time_slot'], prefix='time')
df_processed = pd.concat([df_processed, time_dummies], axis=1)

print("✅ Categorical encoding completed!")

# Prepare features and target
feature_columns = [
    'route_id_encoded', 'distance_km', 'fuel_cost', 'maintenance_cost',
    'no_passengers', 'travel_duration_min', 'time_slot_encoded'
]

X = df_processed[feature_columns]
y = df_processed['total_cost']

print(f"\n📊 Features shape: {X.shape}")
print(f"🎯 Target shape: {y.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n🔄 Training set: {X_train.shape}")
print(f"🔄 Test set: {X_test.shape}")

# ✅ STEP 4: Build and Train ML Model
print("\n" + "="*50)
print("🤖 MACHINE LEARNING MODEL")
print("="*50)

# Initialize and train Random Forest model
print("\n🌲 Training Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5
)

rf_model.fit(X_train, y_train)
print("✅ Model training completed!")

# Make predictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Model evaluation
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\n📊 MODEL PERFORMANCE:")
print(f"📈 Training MSE: {train_mse:.2f}")
print(f"📈 Test MSE: {test_mse:.2f}")
print(f"📈 Training R²: {train_r2:.3f}")
print(f"📈 Test R²: {test_r2:.3f}")

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('🤖 Model Performance Visualization', fontsize=16, fontweight='bold')

# Actual vs Predicted
axes[0].scatter(y_test, y_pred_test, alpha=0.6, color='#1f77b4')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Cost ($)')
axes[0].set_ylabel('Predicted Cost ($)')
axes[0].set_title(f'Actual vs Predicted (R² = {test_r2:.3f})')
axes[0].grid(True, alpha=0.3)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=True)

axes[1].barh(feature_importance['feature'], feature_importance['importance'])
axes[1].set_xlabel('Feature Importance')
axes[1].set_title('Feature Importance Ranking')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n🎯 TOP 3 MOST IMPORTANT FEATURES:")
for i, (feature, importance) in enumerate(feature_importance.tail(3).values, 1):
    print(f"{i}. {feature}: {importance:.3f}")

# ✅ STEP 5: Cost Optimization Insights
print("\n" + "="*50)
print("💡 COST OPTIMIZATION INSIGHTS")
print("="*50)

# Feature impact analysis
print("\n🔍 FEATURE IMPACT ANALYSIS:")
baseline_cost = df['total_cost'].mean()
print(f"📊 Baseline average cost: ${baseline_cost:.2f}")

# Impact of fuel cost changes
fuel_impact = df.groupby(pd.cut(df['fuel_cost'], bins=5))['total_cost'].mean()
print(f"\n⛽ Fuel Cost Impact:")
for range_val, cost in fuel_impact.items():
    print(f"  {range_val}: ${cost:.2f}")

# Most efficient time slots
time_efficiency = df.groupby('time_slot').agg({
    'total_cost': 'mean',
    'travel_duration_min': 'mean',
    'no_passengers': 'mean'
}).round(2)
print(f"\n⏰ Time Slot Efficiency:")
print(time_efficiency)

# Route efficiency analysis
route_efficiency = df.groupby('route_id').agg({
    'total_cost': 'mean',
    'distance_km': 'mean',
    'no_passengers': 'mean'
}).round(2)
route_efficiency['cost_per_km'] = (route_efficiency['total_cost'] / route_efficiency['distance_km']).round(2)
top_efficient_routes = route_efficiency.nsmallest(5, 'cost_per_km')
print(f"\n🛣️ TOP 5 Most Efficient Routes (by cost per km):")
print(top_efficient_routes)

# Optimization recommendations
print(f"\n🎯 OPTIMIZATION RECOMMENDATIONS:")
print("1. 💰 Focus on Night time slots - lowest average cost")
print("2. ⛽ Monitor fuel costs closely - high correlation with total cost")
print("3. 🚌 Optimize passenger capacity to reduce per-passenger costs")
print("4. 📏 Consider route consolidation for shorter, more efficient paths")
print("5. 🔧 Implement predictive maintenance to control maintenance costs")

