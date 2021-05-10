# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:15:43 2021

@author: MH Xu
"""

from datasets import Dataset
from models import GPR, LinearR, RidgeR, LassoR, ENR, SVMR, KNNR, DTR, MLPR

dataset = ['boston', 'diabetes', 'servo', 'friedman1', 'friedman2', 'friedman3']
data_path = {'boston': './Datasets/Boston/', 
             'diabetes': './Datasets/Diabetes/', 
             'servo': './Datasets/Servo/', 
             'friedman1': './Datasets/Friedman#1/', 
             'friedman2': './Datasets/Friedman#2/', 
             'friedman3': './Datasets/Friedman#3/'}

for dataset_name in dataset:
    gpr_result = []
    lr_result = []
    rr_result = []
    lasso_result = []
    enr_result = []
    svr_result = []
    knn_result = []
    dtr_result = []
    mlp_result = []
    Data = Dataset(data_path[dataset_name])
    for ind in range(1,11):
        train_file_name = dataset_name+'_train_' + str(ind)
        test_file_name = dataset_name+'_test_' + str(ind)
        train_data = Data.read_dataset(train_file_name)
        test_data = Data.read_dataset(test_file_name)
        
        gpr = GPR()
        gpr_model = gpr.fit(train_data['X'], train_data['y'])
        y_predict = gpr.predict(gpr_model, test_data['X'])
        gpr_result.append(gpr.mse(test_data['y'], y_predict))
        
        lr = LinearR()
        lr_model = lr.fit(train_data['X'], train_data['y'])
        y_predict = lr.predict(lr_model, test_data['X'])
        lr_result.append(lr.mse(test_data['y'], y_predict))
        
        rr = RidgeR()
        rr_model = rr.fit(train_data['X'], train_data['y'])
        y_predict = rr.predict(rr_model, test_data['X'])
        rr_result.append(rr.mse(test_data['y'], y_predict))
        
        lasso = LassoR()
        lasso_model = lasso.fit(train_data['X'], train_data['y'])
        y_predict = lasso.predict(lasso_model, test_data['X'])
        lasso_result.append(lasso.mse(test_data['y'], y_predict))
        
        enr = ENR()
        enr_model = enr.fit(train_data['X'], train_data['y'])
        y_predict = enr.predict(enr_model, test_data['X'])
        enr_result.append(enr.mse(test_data['y'], y_predict))
        
        svr = SVMR()
        svr_model = svr.fit(train_data['X'], train_data['y'])
        y_predict = svr.predict(svr_model, test_data['X'])
        svr_result.append(svr.mse(test_data['y'], y_predict))
        
        knn = KNNR()
        knn_model = knn.fit(train_data['X'], train_data['y'])
        y_predict = knn.predict(knn_model, test_data['X'])
        knn_result.append(knn.mse(test_data['y'], y_predict))
        
        dtr = DTR()
        dtr_model = dtr.fit(train_data['X'], train_data['y'])
        y_predict = dtr.predict(dtr_model, test_data['X'])
        dtr_result.append(dtr.mse(test_data['y'], y_predict))
        
        mlp = MLPR()
        mlp_model = mlp.fit(train_data['X'], train_data['y'])
        y_predict = mlp.predict(mlp_model, test_data['X'])
        mlp_result.append(mlp.mse(test_data['y'], y_predict))
    # write the results into txt file.
    f = open('./Results/'+dataset_name+'_result.txt','w')
    f.write('——'*22+'\n')
    f.write('|index\t|GPR\t|LR \t|RidgeR\t|LassoR\t|ENR\t|SVR\t|KNN\t|DTR\t|MLP\t|\n')
    f.write('——'*22+'\n')
    for i in range(10):
        f.write('|{}\t|{:.4f}\t|{:.4f}\t|{:.4f}\t|{:.4f}\t|{:.4f}\t|{:.4f}\t|{:.4f}\t|{:.4f}\t|{:.4f}\t|\n'.format(
            int(i+1),
            gpr_result[i],
            lr_result[i],
            rr_result[i],
            lasso_result[i],
            enr_result[i],
            svr_result[i],
            knn_result[i],
            dtr_result[i],
            mlp_result[i]))
    f.write('——'*22+'\n')
    f.write('|avg\t|{:.4f}\t|{:.4f}\t|{:.4f}\t|{:.4f}\t|{:.4f}\t|{:.4f}\t|{:.4f}\t|{:.4f}\t|{:.4f}\t|\n'.format(
        sum(gpr_result)/10,
        sum(lr_result)/10,
        sum(rr_result)/10,
        sum(lasso_result)/10,
        sum(enr_result)/10,
        sum(svr_result)/10,
        sum(knn_result)/10,
        sum(dtr_result)/10,
        sum(mlp_result)/10))
    f.write('——'*22+'\n')
    f.close()