# -*- coding: utf-8 -*-  
# Created by chenlu on May 5nd
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import *
import xgboost as xgb
from sklearn import preprocessing
import pandas as pd

def train_test_split(X):
    train = X[X['day']!=10]
    test = X[X['day']==10]
    return train.drop(['is_attributed'], axis=1), train['is_attributed'],\
           test.drop(['is_attributed'], axis=1), test['is_attributed']
    
def train_dev_split(X_total):
    X_train_ = X_total[(X_total['day']==7)|(X_total['day']==8)]
    X_train = X_train_.drop(['is_attributed'], axis=1)
    y_train = X_train_['is_attributed']
    X_dev_ = X_total[X_total['day']==9]
    X_dev = X_dev_.drop(['is_attributed'], axis=1)
    y_dev = X_dev_['is_attributed']
    return X_train, y_train, X_dev, y_dev

def easy_train(X_total):
    # Params from: https://www.kaggle.com/aharless/swetha-s-xgboost-revised
    param_dist = {
        'max_depth' : 4,
        'subsample' : 0.8,
        'colsample_bytree' : 0.7,
        'colsample_bylevel' : 0.7,
        'scale_pos_weight' : 20,
        'min_child_weight' : 0,
        'reg_alpha' : 4,
        'n_jobs' : 10,
        'n_estimator' : 10, 
        'objective' : 'binary:logistic'
        }

    clf = xgb.XGBClassifier(**param_dist)
    
    X_total_ = X_total.drop(['click_time'], axis=1) if 'click_time' in X_total.columns else X_total
    
    X_train, y_train, X_dev, y_dev = train_dev_split(X_total_)
    clf.fit(X_train, y_train,
            eval_set=[(X_dev, y_dev)], 
            eval_metric='auc',
            verbose=False)
    evals_result = clf.evals_result()
    return clf, evals_result


def xgb_train(X_total):
    # Params from: https://www.kaggle.com/aharless/swetha-s-xgboost-revised
    param_dist = {
        'max_depth' : 4,
        'subsample' : 0.8,
        'colsample_bytree' : 0.7,
        'colsample_bylevel' : 0.7,
        'scale_pos_weight' : 20,
        'min_child_weight' : 0,
        'reg_alpha' : 4,
        'n_jobs' : 4,
        'n_estimator' : 100, 
        'objective' : 'binary:logistic'
        }

    clf = xgb.XGBClassifier(**param_dist)
    
    X_total_ = X_total.drop(['click_time'], axis=1) if 'click_time' in X_total.columns else X_total
    
    X_train, y_train, X_dev, y_dev = train_dev_split(X_total_)
    clf.fit(X_train, y_train,
            eval_set=[(X_dev, y_dev)], 
            eval_metric='auc',
            verbose=False)
    evals_result = clf.evals_result()
    return clf, evals_result

def grid_search_cv(X, y):
    """
    brute force scan for all parameters, here are the tricks
    usually max_depth is 6,7,8
    learning rate is around 0.05, but small changes may make big diff
    tuning min_child_weight subsample colsample_bytree can have 
    much fun of fighting against overfit 
    n_estimators is how many round of boosting
    finally, ensemble xgboost with multiple seeds may reduce variance
    """
    xgb_model = xgb.XGBClassifier()
    parameters = {'nthread':[4], # when use hyperthread, xgboost may become slower
                  'objective':['binary:logistic'], # 概率
                  'learning_rate': [0.15], # 学习率，控制每次迭代更新权重时的步长，默认0.3。值越小，训练越慢。典型值为0.01-0.2。
                  'max_depth': [5], # 值越大，越容易过拟合；值越小，越容易欠拟合。
                  'min_child_weight': [5], # 值越大，越容易欠拟合；值越小，越容易过拟合（值较大时，避免模型学习到局部的特殊样本）。
                  'silent': [1],
                  'subsample': [0.8], # 训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。
                  'colsample_bytree': [0.8], # 训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。
                  'n_estimators': [100], # 总共迭代的次数，即决策树的个数 number of trees, change it to 1000 for better results
                  'missing':[0],
                  'scale_pos_weight':[1],
                  'seed': [1337]}
    cv = GridSearchCV(xgb_model, 
                       parameters, 
                       n_jobs=5, 
                       cv=StratifiedKFold(y, n_folds=3, shuffle=False), 
                       scoring='roc_auc',
                       verbose=0, 
                       refit=True)

    cv.fit(X, y)
    
    return cv.best_estimator_


def plot_feature_importance(clf):
    # Get xgBoost importances
    """
    ‘weight’ - the number of times a feature is used to split the data across all trees. 
    ‘gain’ - the average gain of the feature when it is used in trees 
    ‘cover’ - the average coverage of the feature when it is used in trees
    """
    importance_dict = {}
    for import_type in ['weight', 'gain', 'cover']:
        importance_dict['xgBoost-'+import_type] = clf.get_booster().get_score(importance_type=import_type)

    # MinMax scale all importances
    importance_df = pd.DataFrame(importance_dict).fillna(0)
    importance_df = pd.DataFrame(
        preprocessing.MinMaxScaler().fit_transform(importance_df),
        columns=importance_df.columns,
        index=importance_df.index
    )

    # Create mean column
    importance_df['mean'] = importance_df.mean(axis=1)

    # Plot the feature importances
    importance_df.sort_values('mean').plot(kind='bar', figsize=(20, 7))