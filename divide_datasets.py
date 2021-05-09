# -*- coding: utf-8 -*-
"""
Output all the training set and testing set according to 10-fold cross validation.

@author: MH Xu
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit

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