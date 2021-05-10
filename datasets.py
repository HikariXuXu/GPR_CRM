# -*- coding: utf-8 -*-
"""
Created on Mon May 10 12:46:19 2021

@author: MH Xu
"""

import pandas as pd
import numpy as np

class Dataset():
    def __init__(self, data_path):
        self.data_path = data_path
    
    def data_file_path(self, file_name):
        return '{}{}.txt'.format(self.data_path, file_name)
    
    def read_dataset(self, file_name, header=0, delimiter=','):
        data = pd.read_csv(self.data_file_path(file_name),
                           header=header, delimiter=delimiter)
        X = np.array(data.iloc[:, :-1])
        y = np.array(data.iloc[:, -1])[:,None]
        return {'X':X, 'y':y}
