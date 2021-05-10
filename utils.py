# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:05:05 2021

@author: MH Xu
"""

from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

train_X, train_y = None, None
class BayesianSearchCV():
    def __init__(self, train_X, train_y, cv, method_name):
        self.name = 'Bayesian Optimization for Parameter Tunning'
        self.train_X, self.train_y = train_X, train_y
        self.cv = cv
        self.method_name = method_name
        self.method_f = {'RidgeR': self.RidgeR_evaluate,
                         'LassoR': self.LassoR_evaluate, 
                         'ENR': self.ENR_evaluate,
                         'SVR': self.SVR_evaluate,
                         'KNNR': self.KNN_evaluate,
                         'DTR': self.DTR_evaluate,
                         'MLPR': self.MLPR_evaluate}
    
    def BayesianSearch(self, params, init_points, num_iter=50):
        """Bayesian Optimization"""
        optimizer = BayesianOptimization(self.method_f[self.method_name], params)
        optimizer.maximize(init_points=init_points, n_iter=num_iter)
        params = optimizer.max
        print(params)
        return params

    def RidgeR_evaluate(self, alpha):
        """DIY ridge regression evaluation function"""
    
        # Fixed hyperparameter
        param = {
            'fit_intercept': True, 
            'normalize': False, 
            'copy_X': True, 
            'max_iter': None, 
            'solver': 'auto', 
            'random_state': 0, 
            'tol': 0.001}
    
        # generated hyperparameter by Bayesian Optimizer
        param['alpha'] = float(alpha)
    
        # k-fold cross validation
        val = cross_val_score(Ridge(**param), self.train_X, self.train_y,
                              scoring='neg_mean_squared_error', cv=self.cv).mean()
    
        return val
    
    def LassoR_evaluate(self, alpha):
        """DIY lasso regression evaluation function"""
    
        # Fixed hyperparameter
        param = {
            'fit_intercept': True,
            'normalize': False,
            'precompute': False,
            'copy_X': True,
            'max_iter': 1000,
            'warm_start': False,
            'positive': False,
            'selection': 'cyclic', 
            'random_state': 0, 
            'tol': 0.0001}
    
        # generated hyperparameter by Bayesian Optimizer
        param['alpha'] = float(alpha)
    
        # k-fold cross validation
        val = cross_val_score(Lasso(**param), self.train_X, self.train_y,
                              scoring='neg_mean_squared_error', cv=self.cv).mean()
    
        return val
    
    def ENR_evaluate(self, alpha, l1_ratio):
        """DIY ElasticNet regression evaluation function"""
    
        # Fixed hyperparameter
        param = {
            'fit_intercept': True,
            'normalize': False,
            'precompute': False,
            'max_iter': 1000,
            'copy_X': True,
            'tol': 0.0001,
            'warm_start': False,
            'positive': False,
            'random_state': 0,
            'selection': 'cyclic'}
    
        # generated hyperparameter by Bayesian Optimizer
        param['alpha'] = float(alpha)
        param['l1_ratio'] = float(l1_ratio)
    
        # k-fold cross validation
        val = cross_val_score(ElasticNet(**param), self.train_X, self.train_y,
                              scoring='neg_mean_squared_error', cv=self.cv).mean()
    
        return val
    
    def SVR_evaluate(self, C, gamma):
        """DIY support vector regression evaluation function"""
    
        # Fixed hyperparameter
        param = {
            'kernel': 'rbf',
            'degree': 3,
            'coef0': 0.0,
            'tol': 0.001,
            'epsilon': 0.1,
            'shrinking': True,
            'cache_size': 200,
            'verbose': False,
            'max_iter': - 1}
    
        # generated hyperparameter by Bayesian Optimizer
        param['C'] = float(C)
        param['gamma'] = float(gamma)
    
        # k-fold cross validation
        val = cross_val_score(SVR(**param), self.train_X, self.train_y,
                              scoring='neg_mean_squared_error', cv=self.cv).mean()
    
        return val
    
    def KNN_evaluate(self, n_neighbors):
        """DIY knn regression evaluation function"""
    
        # Fixed hyperparameter
        param = {
            'weights': 'uniform',
            'algorithm': 'auto',
            'leaf_size': 30,
            'p': 2,
            'metric': 'minkowski',
            'metric_params': None,
            'n_jobs': None}
    
        # generated hyperparameter by Bayesian Optimizer
        param['n_neighbors'] = int(n_neighbors)
    
        # k-fold cross validation
        val = cross_val_score(KNeighborsRegressor(**param), self.train_X, self.train_y,
                              scoring='neg_mean_squared_error', cv=self.cv).mean()
    
        return val
    
    def DTR_evaluate(self, max_depth, min_samples_split, min_samples_leaf, max_features):
        """DIY decision tree regressor evaluation function"""
    
        # Fixed hyperparameter
        param = {
            'criterion': 'mse',
            'splitter': 'best',
            'min_weight_fraction_leaf': 0.0,
            'random_state': 0, 
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'min_impurity_split': None,
            'ccp_alpha': 0.0}
    
        # generated hyperparameter by Bayesian Optimizer
        param['max_depth'] = int(max_depth)
        param['min_samples_split'] = float(min_samples_split)
        param['min_samples_leaf'] = float(min_samples_leaf)
        param['max_features'] = int(max_features)
    
        # k-fold cross validation
        val = cross_val_score(DecisionTreeRegressor(**param), self.train_X, self.train_y,
                              scoring='neg_mean_squared_error', cv=self.cv).mean()
    
        return val
    
    def MLPR_evaluate(self, alpha):
        """DIY Multi-layer Perceptron regressor evaluation function"""
    
        # Fixed hyperparameter
        param = {
            'hidden_layer_sizes': 100,
            'activation': 'relu',
            'solver': 'adam',
            'batch_size': 'auto',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'power_t': 0.5,
            'max_iter': 200,
            'shuffle': True,
            'random_state': 0,
            'tol': 0.0001,
            'verbose': False,
            'warm_start': False,
            'momentum': 0.9,
            'nesterovs_momentum': True,
            'early_stopping': False,
            'validation_fraction': 0.1,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-08,
            'n_iter_no_change': 10,
            'max_fun': 15000}
    
        # generated hyperparameter by Bayesian Optimizer
        param['alpha'] = float(alpha)
    
        # k-fold cross validation
        val = cross_val_score(MLPRegressor(**param), self.train_X, self.train_y,
                              scoring='neg_mean_squared_error', cv=self.cv).mean()
    
        return val