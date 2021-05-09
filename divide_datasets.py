# -*- coding: utf-8 -*-
"""
Output all the training set and testing set according to 10-fold cross validation.

@author: MH Xu
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import make_friedman1
from sklearn.datasets import make_friedman2
from sklearn.datasets import make_friedman3

ss = StandardScaler()
cv = ShuffleSplit(n_splits=10, test_size=.1, random_state=0)

# Boston Housing dataset
boston_data = pd.read_csv("./Datasets/Boston/BostonHousing.csv")
boston_data['chas'] = boston_data['chas'].apply(lambda x: int(x))
boston_std_data = ss.fit_transform(boston_data)
count = 1
for train_index, test_index in cv.split(boston_std_data):
    train_set = pd.DataFrame(np.array(boston_std_data)[train_index])
    test_set = pd.DataFrame(np.array(boston_std_data)[test_index])
    train_set.columns = boston_data.columns.values.tolist()
    test_set.columns = boston_data.columns.values.tolist()
    train_set.to_csv('./Datasets/Boston/boston_train_'+str(count)+'.txt', index=False)
    test_set.to_csv('./Datasets/Boston/boston_test_'+str(count)+'.txt', index=False)
    count += 1

# Diabetes dataset
diabetes_data = pd.read_csv("./Datasets/Diabetes/Diabetes.txt", delimiter='\t')
diabetes_std_data = ss.fit_transform(diabetes_data)
count = 1
for train_index, test_index in cv.split(diabetes_std_data):
    train_set = pd.DataFrame(np.array(diabetes_std_data)[train_index])
    test_set = pd.DataFrame(np.array(diabetes_std_data)[test_index])
    train_set.columns = diabetes_data.columns.values.tolist()
    test_set.columns = diabetes_data.columns.values.tolist()
    train_set.to_csv('./Datasets/Diabetes/diabetes_train_'+str(count)+'.txt', index=False)
    test_set.to_csv('./Datasets/Diabetes/diabetes_test_'+str(count)+'.txt', index=False)
    count += 1

# Servo dataset
servo_data = pd.read_csv("./Datasets/Servo/servo.data", header=None)
servo_data.columns = ['motor', 'screw', 'pgain', 'vgain', 'class']
# transform variable motor and screw to dummy variable
servo_data['motor_A'] = servo_data['motor'].apply(lambda x: 1 if x == 'A' else 0)
servo_data['motor_B'] = servo_data['motor'].apply(lambda x: 1 if x == 'B' else 0)
servo_data['motor_C'] = servo_data['motor'].apply(lambda x: 1 if x == 'C' else 0)
servo_data['motor_D'] = servo_data['motor'].apply(lambda x: 1 if x == 'D' else 0)
servo_data['motor_E'] = servo_data['motor'].apply(lambda x: 1 if x == 'E' else 0)
servo_data['screw_A'] = servo_data['screw'].apply(lambda x: 1 if x == 'A' else 0)
servo_data['screw_B'] = servo_data['screw'].apply(lambda x: 1 if x == 'B' else 0)
servo_data['screw_C'] = servo_data['screw'].apply(lambda x: 1 if x == 'C' else 0)
servo_data['screw_D'] = servo_data['screw'].apply(lambda x: 1 if x == 'D' else 0)
servo_data['screw_E'] = servo_data['screw'].apply(lambda x: 1 if x == 'E' else 0)
servo_data = servo_data.drop(['motor', 'screw'], axis = 1)
cols = servo_data.columns.tolist()
cols = cols[:2] + cols[3:] + [cols[2]]
servo_data = servo_data[cols]
servo_std_data = ss.fit_transform(servo_data)
count = 1
for train_index, test_index in cv.split(servo_std_data):
    train_set = pd.DataFrame(np.array(servo_std_data)[train_index])
    test_set = pd.DataFrame(np.array(servo_std_data)[test_index])
    train_set.columns = servo_data.columns.values.tolist()
    test_set.columns = servo_data.columns.values.tolist()
    train_set.to_csv('./Datasets/Servo/servo_train_'+str(count)+'.txt', index=False)
    test_set.to_csv('./Datasets/Servo/servo_test_'+str(count)+'.txt', index=False)
    count += 1

# Friedman#1 dataset
X, y = make_friedman1(n_samples=200, noise=1.0, random_state=0)
friedman1_data = pd.DataFrame(np.column_stack((X, y[:,None])))
friedman1_data.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'y']
friedman1_data.to_csv('./Datasets/Friedman#1/friedman#1.data', index=False)
friedman1_std_data = ss.fit_transform(friedman1_data)
count = 1
for train_index, test_index in cv.split(friedman1_std_data):
    train_set = pd.DataFrame(np.array(friedman1_std_data)[train_index])
    test_set = pd.DataFrame(np.array(friedman1_std_data)[test_index])
    train_set.columns = friedman1_data.columns.values.tolist()
    test_set.columns = friedman1_data.columns.values.tolist()
    train_set.to_csv('./Datasets/Friedman#1/friedman1_train_'+str(count)+'.txt', index=False)
    test_set.to_csv('./Datasets/Friedman#1/friedman1_test_'+str(count)+'.txt', index=False)
    count += 1

# Friedman#2 dataset
X, y = make_friedman2(n_samples=200, noise=3.0, random_state=0)
friedman2_data = pd.DataFrame(np.column_stack((X, y[:,None])))
friedman2_data.columns = ['x1', 'x2', 'x3', 'x4', 'y']
friedman2_data.to_csv('./Datasets/Friedman#2/friedman#2.data', index=False)
friedman2_std_data = ss.fit_transform(friedman2_data)
count = 1
for train_index, test_index in cv.split(friedman2_std_data):
    train_set = pd.DataFrame(np.array(friedman2_std_data)[train_index])
    test_set = pd.DataFrame(np.array(friedman2_std_data)[test_index])
    train_set.columns = friedman2_data.columns.values.tolist()
    test_set.columns = friedman2_data.columns.values.tolist()
    train_set.to_csv('./Datasets/Friedman#2/friedman2_train_'+str(count)+'.txt', index=False)
    test_set.to_csv('./Datasets/Friedman#2/friedman2_test_'+str(count)+'.txt', index=False)
    count += 1

# Friedman#3 dataset
X, y = make_friedman3(n_samples=200, noise=1.0, random_state=0)
friedman3_data = pd.DataFrame(np.column_stack((X, y[:,None])))
friedman3_data.columns = ['x1', 'x2', 'x3', 'x4', 'y']
friedman3_data.to_csv('./Datasets/Friedman#3/friedman#3.data', index=False)
friedman3_std_data = ss.fit_transform(friedman3_data)
count = 1
for train_index, test_index in cv.split(friedman3_std_data):
    train_set = pd.DataFrame(np.array(friedman3_std_data)[train_index])
    test_set = pd.DataFrame(np.array(friedman3_std_data)[test_index])
    train_set.columns = friedman3_data.columns.values.tolist()
    test_set.columns = friedman3_data.columns.values.tolist()
    train_set.to_csv('./Datasets/Friedman#3/friedman3_train_'+str(count)+'.txt', index=False)
    test_set.to_csv('./Datasets/Friedman#3/friedman3_test_'+str(count)+'.txt', index=False)
    count += 1