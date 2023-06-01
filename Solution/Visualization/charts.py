import csv

import pandas as pd
from matplotlib import pyplot as plt

clean_data = []
with open('cleaned_data.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        clean_data.append(row)

clean_dataFrame = pd.DataFrame(clean_data)

unique_items = clean_dataFrame[1].unique()

item_counts = clean_dataFrame[1].value_counts()

plt.bar(unique_items, item_counts)

plt.xlabel('Unique Items')
plt.ylabel('Frequency Count')

plt.show()
