import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

data = pd.read_csv('cleaned_data.csv')

print(data.describe(include='all'))

categorical_columns = ['body_type', 'diet', 'drinks', 'drugs', 'ethnicity', 'job', 'location',
                       'offspring', 'orientation', 'religion', 'sex', 'education']

chi_statistic_results = []

for column in categorical_columns:
    plt.figure(figsize=(10,6))
    sns.countplot(x=column, data=data, order= data[column].value_counts().index)
    plt.title(f'{column} distribution')
    plt.xticks(rotation=90)
    plt.show()

    contingency_table = pd.crosstab(data[column], data['smokes'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-square test for {column} and smokes")
    print(f"Chi2 statistic: {chi2}, p-value: {p}")


# data_encoded = pd.get_dummies(data)
#
# correlations = data_encoded.corr()
# plt.figure(figsize=(10,8))
# sns.heatmap(correlations, annot=True, cmap='coolwarm')
# plt.title('Correlation matrix')
# plt.show()

categories = ['body_type', 'diet', 'drinks', 'drugs', 'ethnicity', 'job', 'location', 'offspring', 'orientation', 'religion', 'sex', 'education']
chi2_statistics = [103.67419073738384, 76.41525587116178, 117.9811239370458, 229.382054255451, 328.5909424445085, 144.00799786030916, 269.6153810526109, 96.74166127329042, 31.460074299610277, 227.2596255765101, 7.7144392043976415, 273.5009588584229]

chi2_matrix = np.array(chi2_statistics).reshape(-1, 1)

plt.figure(figsize=(10, 8))
sns.heatmap(chi2_matrix, annot=True, fmt=".2f", yticklabels=categories, xticklabels=['Chi2'])
plt.title('Chi-square statistics heatmap')
plt.show()
