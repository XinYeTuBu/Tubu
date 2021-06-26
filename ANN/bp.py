# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 13:30:14 2021

@author: ruj72813
"""
import pandas as pd
import pickle
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


dir_path = os.path.dirname(os.getcwd())
read_path = os.path.join(dir_path,'data/data.csv')

data_pd = pd.read_csv(read_path,header=0)
col = ['涂布槽液位','基片定量','传动侧','波美度','打浆度','湿重','涂布率']
train_data = data_pd[col]

X = train_data.values[:,:-1]
y = train_data.values[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)


# =============================================================================
scaler_X = StandardScaler().fit(X_train)
X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)

regress_net = MLPRegressor(hidden_layer_sizes=(15,10),learning_rate='invscaling', solver='lbfgs',\
                            activation="relu",max_iter=20,random_state=0)
# constant, invscaling, adaptive
# lbfgs(quasi-Newton), sgd(随机梯度下降), adam
# 若求解器为lbfg，则不设置minibatch， 设为auto时, batch_size=min(200,n_samples)
# indentity(x), logistic(sigmoid), relu, softmax, tanh


# 训练并得到预测结果
regress_net.fit(X_train,y_train)
y_predict = regress_net.predict(X_test)
y_train_predict = regress_net.predict(X_train)

error1 = mean_absolute_error(y_predict,y_test)
error2 = mean_absolute_error(y_train_predict,y_train)

# 打印误差结果
print("\nerror1:",error1)
print("error2:",error2)
print("all_error:",0.5*error1+0.5*error2)

# 绘出预测部分图像
plt.plot(y_test,'-*')
plt.plot(y_predict,'--o')
plt.legend(['y_test','y_pred'])

# 保存模型
pkl_path = os.path.join(dir_path,'pkl/ann_model.pkl')
ann_pipline ={"scaler_X":scaler_X,"ann":regress_net}
pickle.dump(ann_pipline,open(pkl_path,"wb"))
# =============================================================================
