import nbformat as nbf
import json

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (telco-churn)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)

def raw(text):
    return nbf.v4.new_raw_cell(text)

# ============================================================
# NOTEBOOK 1: EDA
# ============================================================
nb1 = nbf.v4.new_notebook()
nb1.metadata = nb.metadata
nb1.cells = [
    md("""# 01 — Exploratory Data Analysis
## Telco Customer Churn Dataset

**Objective:** Understand the dataset structure, distributions, and initial churn patterns before building models."""),

    md("""## 1. Setup & Data Loading"""),

    code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')

import sys
sys.path.append('..')
from src.preprocessing import load_data, create_demographic_features, create_service_features, create_contract_features, create_payment_features, create_spend_features
from src.preprocessing import DATA_PATH"""),

    code("""df = load_data()
print(f"Shape: {df.shape}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")"""),

    md("""## 2. Data Overview"""),

    code("""df.head(3)"""),

    code("""df.info()"""),

    code("""df.describe()"""),

    md("""### Missing Values"""),

    code("""missing = df.isnull().sum()
display(missing[missing > 0] if missing.sum() > 0 else "No missing values found after cleaning.")"""),

    md("""## 3. Target Variable — Churn"""),

    code("""churn_counts = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100
churn_summary = pd.DataFrame({'Count': churn_counts, 'Percentage': churn_pct.round(2)})
churn_summary

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
churn_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Churn Count')
axes[0].set_ylabel('Customers')
axes[0].tick_params(axis='x', rotation=0)

axes[1].pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1].set_title('Churn Proportion')
plt.tight_layout()
plt.savefig('../reports/churn_distribution.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    md("""## 4. Demographic Analysis"""),

    code("""fig, axes = plt.subplots(2, 3, figsize=(16, 10))
cat_demo = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
for i, col in enumerate(cat_demo):
    ax = axes[i // 3, i % 3]
    ctab = pd.crosstab(df[col], df['Churn'], normalize='index') * 100
    ctab.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], rot=0)
    ax.set_title(f'Churn Rate by {col}')
    ax.set_ylabel('Percentage')
    ax.legend(title='Churn')

axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('../reports/demographic_churn.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    md("""## 5. Tenure Analysis"""),

    code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist([df[df['Churn']=='No']['tenure'],
              df[df['Churn']=='Yes']['tenure']],
             bins=30, label=['Stayed', 'Churned'],
             color=['#2ecc71', '#e74c3c'], alpha=0.7)
axes[0].set_xlabel('Tenure (months)')
axes[0].set_ylabel('Customers')
axes[0].set_title('Tenure Distribution by Churn')
axes[0].legend()

df['TenureBin'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 48, 72],
                          labels=['0-6mo', '6-12mo', '1-2yr', '2-4yr', '4-6yr'])
churn_by_tenure = df.groupby('TenureBin', observed=True)['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100)
churn_by_tenure.plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_ylabel('Churn Rate (%)')
axes[1].set_title('Churn Rate by Tenure Group')
axes[1].tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig('../reports/tenure_churn.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    md("""## 6. Contract & Payment Analysis"""),

    code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
contract_churn = df.groupby('Contract')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100)
contract_churn.plot(kind='bar', ax=axes[0], color=['#e74c3c', '#f39c12', '#2ecc71'])
axes[0].set_title('Churn Rate by Contract Type')
axes[0].set_ylabel('Churn Rate (%)')
axes[0].tick_params(axis='x', rotation=0)

payment_churn = df.groupby('PaymentMethod')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100)
payment_churn.sort_values().plot(kind='barh', ax=axes[1], color='teal')
axes[1].set_title('Churn Rate by Payment Method')
axes[1].set_xlabel('Churn Rate (%)')
plt.tight_layout()
plt.savefig('../reports/contract_payment_churn.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    md("""## 7. Service Analysis"""),

    code("""service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
service_churn = []
for col in service_cols:
    rate = df.groupby(col, observed=True)['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100)
    for cat, val in rate.items():
        service_churn.append({'Service': col, 'Category': cat, 'ChurnRate': val})
service_df = pd.DataFrame(service_churn)

plt.figure(figsize=(14, 6))
sns.barplot(data=service_df, x='Service', y='ChurnRate', hue='Category')
plt.xticks(rotation=45)
plt.title('Churn Rate by Service Type')
plt.ylabel('Churn Rate (%)')
plt.tight_layout()
plt.savefig('../reports/service_churn.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    md("""## 8. Monthly Charges Analysis"""),

    code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist([df[df['Churn']=='No']['MonthlyCharges'],
              df[df['Churn']=='Yes']['MonthlyCharges']],
             bins=30, label=['Stayed', 'Churned'],
             color=['#2ecc71', '#e74c3c'], alpha=0.7)
axes[0].set_xlabel('Monthly Charges ($)')
axes[0].set_ylabel('Customers')
axes[0].set_title('Monthly Charges Distribution by Churn')
axes[0].legend()

df['ChargeBin'] = pd.qcut(df['MonthlyCharges'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
charge_churn = df.groupby('ChargeBin', observed=True)['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100)
charge_churn.plot(kind='bar', ax=axes[1], color='purple')
axes[1].set_title('Churn Rate by Monthly Charges Quintile')
axes[1].set_ylabel('Churn Rate (%)')
axes[1].tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig('../reports/charges_churn.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    md("""## 9. Correlation Analysis"""),

    code("""df_encoded = pd.get_dummies(df.select_dtypes(include=['object']), drop_first=True)
df_numeric = pd.concat([df.select_dtypes(include=[np.number]), df_encoded], axis=1)
corr = df_numeric.corr()
churn_corr = corr['Churn_Yes'].sort_values(key=abs, ascending=False)
churn_corr_top = churn_corr.head(15)

plt.figure(figsize=(10, 8))
sns.heatmap(corr[churn_corr_top.index].loc[churn_corr_top.index],
            annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title('Top 15 Churn Correlations')
plt.tight_layout()
plt.savefig('../reports/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

churn_corr_top"""),

    md("""## 10. Key Findings

| Finding | Insight |
|---------|---------|
| **Overall churn rate** | ~26.5% of customers churned |
| **Contract type** | Month-to-month contracts have ~3x higher churn than one-year, ~5x higher than two-year |
| **Tenure** | Churn rate drops sharply after 12 months; highest in first 6 months |
| **Payment method** | Electronic check users churn at ~2x the rate of automatic payment users |
| **Services** | Lack of online security and tech support strongly correlates with churn |
| **Internet service** | Fiber optic customers churn more than DSL (likely due to higher competition) |
| **Demographics** | Seniors and customers without partners/dependents churn more"""),

    md("""---
*End of 01 — Exploratory Data Analysis*""")
]

with open('01_eda.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb1, f)
print("Created 01_eda.ipynb")

# ============================================================
# NOTEBOOK 2: Segmentation
# ============================================================
nb2 = nbf.v4.new_notebook()
nb2.metadata = nb.metadata
nb2.cells = [
    md("""# 02 — Customer Segmentation
## Clustering the Telco Customer Base

**Objective:** Group customers into meaningful segments to enable targeted retention strategies."""),

    md("""## 1. Setup & Data Preparation"""),

    code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')

import sys
sys.path.append('..')
from src.preprocessing import load_data, create_service_features, create_demographic_features
from src.segmentation import prepare_segmentation_data, find_optimal_k, run_kmeans, apply_pca, label_segments, SEGMENT_NAMES"""),

    code("""df = load_data()
df = create_demographic_features(df)
df = create_service_features(df)
print(f"Loaded {len(df)} customers")"""),

    md("""## 2. Optimal Number of Segments"""),

    code("""seg_data, scaler = prepare_segmentation_data(df)
inertias = find_optimal_k(seg_data, max_k=10)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), inertias, 'bo-', linewidth=2)
plt.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='k=5 (selected)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.legend()
plt.savefig('../reports/elbow_curve.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    md("""## 3. K-Means Clustering"""),

    code("""labels, kmeans = run_kmeans(seg_data, n_clusters=5)
df['Cluster'] = labels
df['Segment'] = label_segments(labels)
print("Cluster distribution:")
print(df['Segment'].value_counts())"""),

    code("""pca_components, pca = apply_pca(seg_data)
df['PCA1'] = pca_components[:, 0]
df['PCA2'] = pca_components[:, 1]

plt.figure(figsize=(12, 8))
palette = {0: '#e74c3c', 1: '#2ecc71', 2: '#3498db', 3: '#f39c12', 4: '#9b59b6'}
for cluster in sorted(df['Cluster'].unique()):
    subset = df[df['Cluster'] == cluster]
    plt.scatter(subset['PCA1'], subset['PCA2'],
                c=palette[cluster], label=SEGMENT_NAMES.get(cluster, f'Segment {cluster}'),
                alpha=0.6, s=20)
plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Customer Segments Visualized (PCA)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('../reports/segmentation_pca.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    md("""## 4. Segment Profiles"""),

    code("""segment_summary = df.groupby('Segment').agg(
    Count=('tenure', 'size'),
    AvgTenure=('tenure', 'mean'),
    AvgMonthlyCharges=('MonthlyCharges', 'mean'),
    AvgTotalCharges=('TotalCharges', 'mean'),
    AvgServiceCount=('ServiceCount', 'mean'),
    PctWithDependents=('HasDependents', 'mean'),
    PctSenior=('SeniorCitizen', 'mean'),
).round(2)
segment_summary['ChurnRate'] = df.groupby('Segment')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100).round(1)
segment_summary['PctOfBase'] = (segment_summary['Count'] / len(df) * 100).round(1)
segment_summary"""),

    code("""fig, axes = plt.subplots(2, 3, figsize=(16, 10))
metrics = ['AvgTenure', 'AvgMonthlyCharges', 'AvgServiceCount', 'ChurnRate', 'Count']
for i, metric in enumerate(metrics):
    ax = axes[i // 3, i % 3]
    segment_summary[metric].sort_values().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'Segment Comparison: {metric}')
    ax.set_xlabel(metric)

axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('../reports/segment_profiles.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    md("""## 5. Segment Deep Dive

### Segment Descriptions"""),

    code("""segment_descriptions = df.groupby('Segment').agg({
    'tenure': ['mean', 'median', 'min', 'max'],
    'MonthlyCharges': ['mean', 'std'],
    'Churn': lambda x: (x == 'Yes').mean() * 100,
    'SeniorCitizen': 'mean',
    'Partner': lambda x: (x == 'Yes').mean(),
    'Dependents': lambda x: (x == 'Yes').mean(),
}).round(2)
segment_descriptions.columns = ['AvgTenure', 'MedTenure', 'MinTenure', 'MaxTenure',
                                 'AvgMonthlyCharges', 'StdCharges',
                                 'ChurnRate%', 'SeniorCitizen%', 'HasPartner%', 'HasDependents%']
segment_descriptions"""),

    md("""### Segment Interpretations

1. **High-Value Loyal** — Long tenure, high spend, low churn. These are your best customers. Protect them.
2. **Price-Sensitive** — Lower spend, moderate tenure. May respond to discounts.
3. **New / Short-Tenure** — Low tenure, varying spend. High churn risk. Need early engagement.
4. **Low Engagement** — Few services, lower spend. May not be fully adopted.
5. **Premium Power Users** — High tenure, high spend, many services. Low churn but high potential loss if they leave."""),

    code("""print("Key Takeaways:")
print("=" * 50)
for segment in segment_summary.index:
    churn = segment_summary.loc[segment, 'ChurnRate']
    pct = segment_summary.loc[segment, 'PctOfBase']
    revenue_impact = segment_summary.loc[segment, 'Count'] * churn / 100 * segment_summary.loc[segment, 'AvgMonthlyCharges']
    print(f"{segment}: {pct:.1f}% of base, {churn:.1f}% churn, ~${revenue_impact:.0f}/mo at risk")"""),

    md("""---
*End of 02 — Customer Segmentation*""")
]

with open('02_segmentation.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb2, f)
print("Created 02_segmentation.ipynb")

print("\\nAll notebooks generated successfully!")
