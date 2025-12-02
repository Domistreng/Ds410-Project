import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the data
df = pd.read_csv('../data/processed/engineeredParams/part-00000-8480a262-db60-42e9-80f1-2a6fa0c73693-c000.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
df.head()

engineered_features = ['day_of_week','rush_hour','is_weekend','is_clear','is_rainy','is_foggy','is_snowy','is_highway','is_local','state_encoded','high_accident_state']
categorical_features = ['day_of_week', 'state_encoded']
binary_features = ['rush_hour','is_weekend','is_clear','is_rainy','is_foggy','is_snowy','is_highway','is_local','high_accident_state']

available_features = [f for f in engineered_features if f in df.columns]
feature_df = df[available_features + ['Severity']].copy()

for col in feature_df.columns:
    if col == 'Severity':
        continue
    if feature_df[col].dtype in ['int64', 'float64']:
        feature_df[col].fillna(feature_df[col].median(), inplace=True)
    else:
        feature_df[col].fillna(feature_df[col].mode()[0] if len(feature_df[col].mode()) > 0 else 'Unknown', inplace=True)
        
le_dict = {}
for col in categorical_features:
    if col in feature_df.columns:
        le = LabelEncoder()
        feature_df[col] = le.fit_transform(feature_df[col].astype(str))
        le_dict[col] = le

for col in binary_features:
    if col in feature_df.columns:
        if feature_df[col].dtype == 'object':
            feature_df[col] = feature_df[col].astype(int)
            
if not os.path.exists('ENGINEERED_FEAT: Feature Correlations with Traffic Severity.png'):

    correlations = feature_df.corr()['Severity'].sort_values(ascending=False)
    correlations = correlations.drop('Severity')  # Remove self-correlation

    print("Correlation with Severity (sorted by absolute value):")
    print("=" * 60)
    for feature, corr in correlations.items():
        print(f"{feature:30s}: {corr:7.4f}")

    # Visualize correlations
    plt.figure(figsize=(10, 8))
    correlations_sorted = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
    colors = ['red' if x < 0 else 'green' for x in correlations_sorted.values]
    plt.barh(range(len(correlations_sorted)), correlations_sorted.values, color=colors)
    plt.yticks(range(len(correlations_sorted)), correlations_sorted.index)
    plt.xlabel('Correlation with Severity')
    plt.title('Feature Correlations with Traffic Severity')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig("ENGINEERED_FEAT: Feature Correlations with Traffic Severity.png")
    plt.close()
    
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

categorical_available = [f for f in categorical_features if f in df.columns]

if len(categorical_available) > 0:
    # Take first 4 categorical features
    top_categorical = categorical_available[:4]
    
    if not os.path.exists('ENGINEERED_FEAT: Categorical Parameters Impact on Severity Distribution.png'):
    
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()

        for idx, feature in enumerate(top_categorical):
            # Calculate severity distribution for each category
            category_severity = df.groupby([feature, 'Severity']).size().unstack(fill_value=0)
            category_severity_pct = category_severity.div(category_severity.sum(axis=1), axis=0) * 100

            # Plot stacked bar chart
            category_severity_pct.plot(kind='bar', stacked=True, ax=axes[idx], 
                                       color=colors[:len(category_severity_pct.columns)], 
                                       alpha=0.8, width=0.8)

            axes[idx].set_title(f'{feature} Impact on Severity', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Percentage (%)')
            axes[idx].legend(title='Severity', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')

        plt.suptitle('Categorical Parameters Impact on Severity Distribution', 
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('ENGINEERED_FEAT: Categorical Parameters Impact on Severity Distribution.png')
        plt.close()

binary_available = [f for f in binary_features if f in df.columns]

if len(binary_available) > 0:
    # Take top 6 binary features
    top_binary = binary_available[:9]
    
    if not os.path.exists('ENGINEERED_FEAT: Binary Features Impact on Average Severity.png'):
    
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, feature in enumerate(top_binary):
            # Calculate average severity for each binary value
            binary_severity = df.groupby(feature)['Severity'].agg(['mean', 'std', 'count']).reset_index()
            binary_severity[feature] = binary_severity[feature].astype(str)

            # Create bar plot
            bars = axes[idx].bar(binary_severity[feature], binary_severity['mean'], 
                                yerr=binary_severity['std'], capsize=5, 
                                color=colors[:len(binary_severity)], alpha=0.7)

            axes[idx].set_title(f'{feature} Impact on Severity', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel(f'{feature} (0=No, 1=Yes)')
            axes[idx].set_ylabel('Average Severity')
            axes[idx].grid(True, alpha=0.3, axis='y')

            # Add count labels
            for bar, count in zip(bars, binary_severity['count']):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                              f'n={count}', ha='center', va='bottom', fontsize=8)

        plt.suptitle('Binary Features Impact on Average Severity', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('ENGINEERED_FEAT: Binary Features Impact on Average Severity.png')
        plt.close()