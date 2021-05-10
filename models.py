# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:28:45 2021

@author: MH Xu
"""

from utils import BayesianSearchCV

import numpy as np
import GPy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

# Model parameter tunning
def ModelOptimization(model, params, train_x, train_y):
    best_params = []
    for param in params:
        cv = GridSearchCV(estimator = model,#模型
                          param_grid = param, #参数列表
                          scoring = "neg_log_loss", #评分规则
                          cv = 3,# 交叉验证次数 
                          n_jobs = -1, #
                         )
        cv.fit(train_x, train_y)
        best_params.append(cv.best_params_)
    return best_params

class GPR():
    def __init__(self):
        self.name = 'Gaussian Processes Regression'
    
    def fit(self, X, y, var=1.0, lengthscale=1.0):
        D = X.shape[1]
        kernel = GPy.kern.RBF(D, var, lengthscale)
        model = GPy.models.GPRegression(X, y, kernel)
        model.constrain_positive('')
        model.optimize()
        return model
    
    def predict(self, model, X):
        posterior_mean, posterior_var = model.predict(X)
        return posterior_mean
    
    def mse(self, y, y_predict):
        print('MSE={:.4f}'.format(mean_squared_error(y, y_predict)))
        return mean_squared_error(y, y_predict)


class LinearR():
    def __init__(self):
        self.name = 'Linear Regression'
    
    def fit(self, X, y):
        lr = LinearRegression()
        model = lr.fit(X, y)
        return model
    
    def predict(self, model, X):
        return model.predict(X)
    
    def mse(self, y, y_predict):
        print('MSE={:.4f}'.format(mean_squared_error(y, y_predict)))
        return mean_squared_error(y, y_predict)


class RidgeR():
    def __init__(self):
        self.name = 'Ridge Regression'
    
    def parameter_tunning(self, train_X, train_y, adj_params=None, cv=10, 
                          init_points=5, num_iter=50):
        ''' Parameter tunning by Bayesian Optimizer.'''
        bayes = BayesianSearchCV(train_X, train_y, cv, 'RidgeR')
        if adj_params == None:
            adj_params = {'alpha': (1e-4,10)}
        tuned_params = bayes.BayesianSearch(adj_params, 
                                            init_points=init_points, 
                                            num_iter=num_iter)
        return tuned_params['params']
    
    def fit(self, X, y, params=None):
        if params == None:
            params = self.parameter_tunning(X, y)
        rr = Ridge(**params)
        model = rr.fit(X, y)
        return model
    
    def predict(self, model, X):
        return model.predict(X)
    
    def mse(self, y, y_predict):
        print('MSE={:.4f}'.format(mean_squared_error(y, y_predict)))
        return mean_squared_error(y, y_predict)


class LassoR():
    def __init__(self):
        self.name = 'Lasso Regression'
    
    def parameter_tunning(self, train_X, train_y, adj_params=None, cv=10, 
                          init_points=5, num_iter=50):
        ''' Parameter tunning by Bayesian Optimizer.'''
        bayes = BayesianSearchCV(train_X, train_y, cv, 'LassoR')
        if adj_params == None:
            adj_params = {'alpha': (1e-4,10)}
        tuned_params = bayes.BayesianSearch(adj_params, 
                                            init_points=init_points, 
                                            num_iter=num_iter)
        return tuned_params['params']
    
    def fit(self, X, y, params=None):
        if params == None:
            params = self.parameter_tunning(X, y)
        lasso = Lasso(**params)
        model = lasso.fit(X, y)
        return model
    
    def predict(self, model, X):
        return model.predict(X)
    
    def mse(self, y, y_predict):
        print('MSE={:.4f}'.format(mean_squared_error(y, y_predict)))
        return mean_squared_error(y, y_predict)


class ENR():
    def __init__(self):
        self.name = 'ElasticNet Regression'
    
    def parameter_tunning(self, train_X, train_y, adj_params=None, cv=10, 
                          init_points=5, num_iter=50):
        ''' Parameter tunning by Bayesian Optimizer.'''
        bayes = BayesianSearchCV(train_X, train_y, cv, 'ENR')
        if adj_params == None:
            adj_params = {'alpha': (1e-4,10), 'l1_ratio': (1e-5,1)}
        tuned_params = bayes.BayesianSearch(adj_params, 
                                            init_points=init_points, 
                                            num_iter=num_iter)
        return tuned_params['params']
    
    def fit(self, X, y, params=None):
        if params == None:
            params = self.parameter_tunning(X, y)
        enr = ElasticNet(**params)
        model = enr.fit(X, y)
        return model
    
    def predict(self, model, X):
        return model.predict(X)
    
    def mse(self, y, y_predict):
        print('MSE={:.4f}'.format(mean_squared_error(y, y_predict)))
        return mean_squared_error(y, y_predict)


class SVMR():
    def __init__(self):
        self.name = 'Support Vector Regression'
    
    def parameter_tunning(self, train_X, train_y, adj_params=None, cv=10, 
                          init_points=5, num_iter=50):
        ''' Parameter tunning by Bayesian Optimizer.'''
        bayes = BayesianSearchCV(train_X, train_y, cv, 'SVR')
        if adj_params == None:
            adj_params = {'C': (1e-4,1000), 'gamma': (1e-8,0.1)}
        tuned_params = bayes.BayesianSearch(adj_params, 
                                            init_points=init_points, 
                                            num_iter=num_iter)
        return tuned_params['params']
    
    def fit(self, X, y, params=None):
        y = np.squeeze(y)
        if params == None:
            params = self.parameter_tunning(X, y)
        svr = SVR(**params)
        model = svr.fit(X, y)
        return model
    
    def predict(self, model, X):
        return model.predict(X)
    
    def mse(self, y, y_predict):
        print('MSE={:.4f}'.format(mean_squared_error(y, y_predict)))
        return mean_squared_error(y, y_predict)


class KNNR():
    def __init__(self):
        self.name = 'KNN Regression'
    
    def parameter_tunning(self, train_X, train_y, adj_params=None, cv=10, 
                          init_points=5, num_iter=50):
        ''' Parameter tunning by Bayesian Optimizer.'''
        bayes = BayesianSearchCV(train_X, train_y, cv, 'KNNR')
        if adj_params == None:
            adj_params = {'n_neighbors': (1,11)}
        tuned_params = bayes.BayesianSearch(adj_params, 
                                            init_points=init_points, 
                                            num_iter=num_iter)
        tuned_params['params']['n_neighbors'] = int(tuned_params['params']['n_neighbors'])
        return tuned_params['params']
    
    def fit(self, X, y, params=None):
        if params == None:
            params = self.parameter_tunning(X, y)
        knn = KNeighborsRegressor(**params)
        model = knn.fit(X, y)
        return model
    
    def predict(self, model, X):
        return model.predict(X)
    
    def mse(self, y, y_predict):
        print('MSE={:.4f}'.format(mean_squared_error(y, y_predict)))
        return mean_squared_error(y, y_predict)


class DTR():
    def __init__(self):
        self.name = 'Decision Tree Regressor'
    
    def parameter_tunning(self, train_X, train_y, adj_params=None, cv=10, 
                          init_points=5, num_iter=50):
        ''' Parameter tunning by Bayesian Optimizer.'''
        bayes = BayesianSearchCV(train_X, train_y, cv, 'DTR')
        if adj_params == None:
            adj_params = {'max_depth': (1,32), 
                          'min_samples_split': (0.1,1),
                          'min_samples_leaf': (1e-3,0.5),
                          'max_features': (1,train_X.shape[1])}
        tuned_params = bayes.BayesianSearch(adj_params, 
                                            init_points=init_points, 
                                            num_iter=num_iter)
        tuned_params['params']['max_depth'] = int(tuned_params['params']['max_depth'])
        tuned_params['params']['max_features'] = int(tuned_params['params']['max_features'])
        return tuned_params['params']
    
    def fit(self, X, y, params=None):
        if params == None:
            params = self.parameter_tunning(X, y)
        dtr = DecisionTreeRegressor(**params)
        model = dtr.fit(X, y)
        return model
    
    def predict(self, model, X):
        return model.predict(X)
    
    def mse(self, y, y_predict):
        print('MSE={:.4f}'.format(mean_squared_error(y, y_predict)))
        return mean_squared_error(y, y_predict)


class MLPR():
    def __init__(self):
        self.name = 'Multi-layer Perceptron Regressor'
    
    def fit(self, X, y, max_iter=500):
        y = np.squeeze(y)
        mlp = MLPRegressor(max_iter=max_iter)
        model = mlp.fit(X, y)
        return model
    
    def predict(self, model, X):
        return model.predict(X)
    
    def mse(self, y, y_predict):
        print('MSE={:.4f}'.format(mean_squared_error(y, y_predict)))
        return mean_squared_error(y, y_predict)