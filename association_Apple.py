import pandas as pd
import pandas_datareader as pdr
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import random
import matplotlib.pyplot as plt

dataset = [['Apple', 'Beer', 'Rice', 'Chicken'],
           ['Apple', 'Beer', 'Rice'],
           ['Apple', 'Beer'],
           ['Apple', 'Bananas'],
           ['Milk', 'Beer', 'Rice', 'Chicken'],
           ['Milk', 'Beer', 'Rice'],
           ['Milk', 'Beer'],
           ['Apple', 'Bananas']]

pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 100)
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
# rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=1)
# print(te_ary.astype("int"))
print(rules)
print(rules[ (rules['lift'] >= 1.33) &
       (rules['confidence'] >= 0.8) ])

support=rules.as_matrix(columns=['support'])
confidence=rules.as_matrix(columns=['confidence'])

# for i in range(len(support)):
#     support[i] = support[i] + 0.0025 * (random.randint(1, 10) - 5)
#     confidence[i] = confidence[i] + 0.0025 * (random.randint(1, 10) - 5)

#
# plt.scatter(support, confidence, alpha=0.5, marker="*")
# plt.xlabel('support')
# plt.ylabel('confidence')
# plt.show()
