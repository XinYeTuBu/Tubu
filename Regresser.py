import pickle
import sqlite3
# import numpy as np

# db_path = 'db/records.db'

class ANN(object):

    def __init__(self):
        self.Transform = None
        self.model = None
    
    def load_model(self,folder_path='pkl',model_name='ann_model'):
        filepath = folder_path +'/'+model_name +'.pkl'
        pipline = pickle.load(open(filepath,'rb')) # {names:model}
        self.model = pipline['ann'] 
        self.Transform = pipline['scaler_X'] 
        print("加载ANN模型···")

    def predict(self,X):
        x_test = self.Transform.transform(X)
        y_predict = self.model.predict(x_test)
        return y_predict


class Ensemble():
    
    def __init__(self):
        self.models = None
        self.w = [0.35,0.35,0.3]
    
    def load_model(self,folder_path='pkl',model_name='ensemble_model'):
        filepath = folder_path + '/' + model_name + '.pkl'
        models = pickle.load(open(filepath,'rb'))
        self.models = models['models']
        self.w = models['w']
    
    def predict(self,X_test):
        self.load_model()
        models = self.models
        y_predict = 0
        for i, model in enumerate(models.values()):
            y_predict_temp = model.predict(X_test)
            y_predict += self.w[i]*y_predict_temp
        return y_predict
    
        