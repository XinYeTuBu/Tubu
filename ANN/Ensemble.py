# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:13:36 2021

@author: ruj72813
"""

import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet, \
     TheilSenRegressor, HuberRegressor, RANSACRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import itertools
import pickle
import os

dir_path = os.path.dirname(os.getcwd())
read_path = os.path.join(dir_path,'data/data.csv')
pkl_path = os.path.join(dir_path,'pkl/ensemble_model.pkl')


def train_Ensemble(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, \
                                                        test_size=0.2, random_state=1)
    regs = [
        ['Lasso', Lasso()],
        ['LinearRegression', LinearRegression()],
        ['Ridge', Ridge()],
        ['ElasticNet', ElasticNet()],
        ['TheilSenRegressor', TheilSenRegressor()],
        ['RANSACRegressor', RANSACRegressor()],
        ['HuberRegressor', HuberRegressor()],
        ['SVR',SVR(kernel='linear')],
        ['DecisionTreeRegressor', DecisionTreeRegressor()],
        ['ExtraTreeRegressor', ExtraTreeRegressor()],
        ['AdaBoostRegressor', AdaBoostRegressor(n_estimators=6)],
        ['ExtraTreesRegressor', ExtraTreesRegressor(n_estimators=6)],
        ['GradientBoostingRegressor', GradientBoostingRegressor(n_estimators=6)],
        ['RandomForestRegressor', RandomForestRegressor(n_estimators=6)],
        ['XGBRegressor', XGBRegressor(n_estimators=6, )],
    ]

    preds = []
    for reg_name, reg in regs:
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        score = mean_absolute_error(y_test, y_pred)
        preds.append([reg_name, y_pred])
    # 对模型做各种组合寻找最优的方案
    final_results = []
    score_dict = {}
    for comb_length in range(1, len(regs) + 1):
        #    print('Model Amount :',comb_length)
        results = []
        for comb in itertools.combinations(preds, comb_length):
            pred_sum = 0
            model_name = []
            for reg_name, pred in comb:
                pred_sum += pred
                model_name.append(reg_name)
            pred_sum /= comb_length
            model_name = '+'.join(model_name)
            score = mean_absolute_error(y_test, pred_sum)
            results.append([model_name, score])
        score_dict[comb[0][0]] = score
        results = sorted(results, key=lambda x: x[1])
        final_results.append(results[0])

    final_results = sorted(final_results, key=lambda x: x[1])
    # 保存较好的模型

    return final_results[0], regs, y_test, preds, score_dict


def save_model(x, y, file_path=pkl_path):
    # regs:dict{decision}; preds:predict(decision);
    final_results, regs, y_test, preds, score_dict = train_Ensemble(x, y)
    print("the error of Ensemble model is:", final_results[1])
    best_model_names = final_results[0]
    model_names = best_model_names.split('+')
    all_model_names = [x[0] for x in regs]  # 所有模型的名字
    models = {}
    w = []
    for model_name in model_names:
        index = all_model_names.index(model_name)
        models[model_name] = regs[index][1]
        w.append(score_dict[model_name])
        # print(models[model_name])
    w = w / sum(w)
    ensemble_pipline = {"models": models, "w": w}
    pickle.dump(ensemble_pipline, open(file_path, 'wb'), -1)
    return model_names, y_test


def predict(X_test, weights=[0.35, 0.3, 0.35], file_path=pkl_path):
    ensemble_pipline = pickle.load(open(file_path, 'rb'))  # {names:model}
    models = ensemble_pipline["models"]
    weights = ensemble_pipline["w"]
    y_predict = 0
    y_predict_detail = []

    for i, model in enumerate(models.values()):
        y_predict_temp = model.predict(X_test)
        y_predict += weights[i] * y_predict_temp
        y_predict_detail.append(y_predict_temp)
    return y_predict


if __name__ == "__main__":
    # 读取数据
    data_pd = pd.read_csv(read_path,header=0)
    col = ['涂布槽液位','基片定量','传动侧','波美度','打浆度','湿重','涂布率']
    train_data = data_pd[col]
    
    X = train_data.values[:,:-1]
    y = train_data.values[:,-1]
    
    model = save_model(X, y)


    # 绘图预测
    import matplotlib.pyplot as plt
    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                    test_size=0.2, random_state=1)
    plt.plot(predict(X_test),'--*')
    plt.plot(y_test,'-o')
    plt.legend(['y_pre','y_tst'])
    