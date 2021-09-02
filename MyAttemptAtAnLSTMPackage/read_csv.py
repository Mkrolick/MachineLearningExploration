import pandas as pd
import numpy as np
from MyAttemptAtAnLSTMPackage.classes import Network
df = pd.read_csv("/Users/malcolmkrolick/Documents/GitHub/MachineLearningExploration/MyAttemptAtAnLSTMPackage/mnist_test.csv")
print(df.info())
#print(df.head())
#print(df.loc[0])
"""
for value, item in enumerate(df.loc[0]):
    if item != 0:
        print(item, value)
"""
#print(df.columns)
#print(df.columns)

label_dataset = df['label']
#print(df.iloc[0])


train_data = []

for i in range(len(df.columns)):
    #print(np.array(df.iloc[i]))
    train_data.append(np.array(df.iloc[i]))


new_label_data_set = []
for label, values in zip(label_dataset, train_data):

    """
    temp_list = []
    for i in range(label):
        temp_list.append(0)
    temp_list.append(1)
    while len(temp_list) <= 9:
        temp_list.append(0)

    #Done in 3.258 seconds, 3.631, 3.23
    # avg = 3.373
    """



    temp_list = np.zeros(10, dtype="int")
    temp_list[label] = 1
    label = temp_list
    new_label_data_set.append(label)
    #Done in 3.379 seconds, 3.406, 3.2
    # avg = 3.283

    print(label, values)


Network = Network()
