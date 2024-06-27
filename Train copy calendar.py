import optuna
from optuna.samplers import TPESampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time,re,warnings
from tqdm import tqdm
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso,Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")

#读取数据
stockname='000333'
start_time = time.time()

folder_path = "/data/work/yangsq/trainset_all/"+stockname+"/calendar/"  # 替换为文件夹的路径
# 获取文件夹中的所有文件名
file_names = os.listdir(folder_path)
sorted_file_names = sorted(file_names, key=lambda x: int(x.split('trainset')[1].split('.')[0]))
datedf=pd.read_csv('/data/work/yangsq/date.csv')
all_date=datedf[stockname].tolist()

readdict={}
for i in range(len(sorted_file_names)):
    temp=pd.read_pickle("/data/work/yangsq/trainset_all/"+stockname+"/calendar/%s" %sorted_file_names[i])
    readdict[all_date[i]]=temp
trainset=pd.concat(readdict)
trainset=trainset.loc[np.isfinite(trainset.T).all()]

trainset=trainset.reset_index()
new_df = pd.concat([trainset.iloc[:, :2], trainset.iloc[:, -2:], pd.DataFrame(columns=['predl1', 'predl2', 'predrf1', 'predrf2'])], axis=1)
new_df.set_index(new_df.columns[0], inplace=True, drop=True)
print(trainset)
trainset=trainset.set_index('level_0')
trainset=trainset.drop(trainset.columns[0], axis=1)
trainset = trainset.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '_', str(x)))
# print(trainset)

Xset=trainset.iloc[:,:108]
yset1=trainset.iloc[:,108]
yset2=trainset.iloc[:,109]
all_date=trainset.index.unique().sort_values()
feature_names = Xset.columns

# 每天训练出的模型的特征重要性记录字典'rf','lasso','lgb'
feature_importance = {'rf':pd.DataFrame(index = all_date[20:],columns = feature_names),'lasso':pd.DataFrame(index = all_date[20:],columns = feature_names),'lgb':pd.DataFrame(index = all_date[20:],columns = feature_names)}

end_time = time.time(); total_time = end_time - start_time
print(f"read_pickle completed! Used {total_time}s.\nlen(all_date) = ",len(all_date))
print("---Training/Hyper Parameter Tuning starts:---")
if len(all_date)<5:
    raise ValueError
today_index=0


#封装的train_model函数
def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)

#封装的数据处理函数
def process_data(Xset, yset1, yset2, today_index, i):
    X_train=Xset.loc[all_date[(today_index+i*5):(today_index+(i+1)*5)]]
    y_train1=yset1.loc[all_date[(today_index+i*5):(today_index+(i+1)*5)]]
    y_train2=yset2.loc[all_date[(today_index+i*5):(today_index+(i+1)*5)]]
    
    X_test=Xset.loc[all_date[(today_index+(i+1)*5):(today_index+(i+2)*5)]]
    y_test1=yset1.loc[all_date[(today_index+(i+1)*5):(today_index+(i+2)*5)]]
    y_test2=yset2.loc[all_date[(today_index+(i+1)*5):(today_index+(i+2)*5)]]
    
    #标准化
    scaler=StandardScaler().fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    return X_train, y_train1, y_train2, X_test, y_test1, y_test2

#调超参数的函数
def RFscore(trial):
    global today_index
    x = trial.suggest_categorical('x', [3, 4, 5, 6, 7])
    count=0
    for i in range(3):
        X_train, y_train1, y_train2, X_test, y_test1, y_test2 = process_data(Xset, yset1, yset2, today_index, i)
        clf_rf=RandomForestRegressor(n_estimators=50,max_samples=0.01,max_depth=x,random_state=42) ###
        count += train_model(clf_rf, X_train, y_train1, X_test, y_test1)
        count += train_model(clf_rf, X_train, y_train2, X_test, y_test2)
    return count

def Lassoscore(trial):
    global today_index
    x = trial.suggest_categorical('x', [10**i for i in range(-8,7)])
    count=0
    for i in range(3):
        X_train, y_train1, y_train2, X_test, y_test1, y_test2 = process_data(Xset, yset1, yset2, today_index, i)
        lasso=Lasso(alpha=x) ###
        count += train_model(lasso, X_train, y_train1, X_test, y_test1)
        count += train_model(lasso, X_train, y_train2, X_test, y_test2)
    return count

def Ridgescore(trial):
    global today_index
    x = trial.suggest_categorical('x', [10**i for i in range(-8,7)])
    count=0
    for i in range(3):
        X_train, y_train1, y_train2, X_test, y_test1, y_test2 = process_data(Xset, yset1, yset2, today_index, i)
        ridge=Ridge(alpha=x) ###
        count += train_model(ridge, X_train, y_train1, X_test, y_test1)
        count += train_model(ridge, X_train, y_train2, X_test, y_test2)
    return count

def Lightscore(trial):
    global today_index
    x = trial.suggest_categorical('x', [2**i-1 for i in range(2,9)])    
    count=0
    for i in range(3):
        X_train, y_train1, y_train2, X_test, y_test1, y_test2 = process_data(Xset, yset1, yset2, today_index, i)
        lgbm=LGBMRegressor(n_estimators=50, subsample=0.5,num_leaves =x,random_state=42,n_jobs=40) ###
        count += train_model(lgbm, X_train, y_train1, X_test, y_test1)
        count += train_model(lgbm, X_train, y_train2, X_test, y_test2)
    return count

def get_best_param(score_func, n_trials=5, timeout=600,n_jobs=1):
    if score_func.__name__ == 'RFscore':
        searchspace = {'x': [3, 4, 5, 6, 7]}
    elif score_func.__name__ == 'Lassoscore':
        searchspace = {'x': [10**i for i in range(-8,7)]}
    elif score_func.__name__ == 'Ridgescore':
        searchspace = {'x': [10**i for i in range(-8,7)]}
    elif score_func.__name__ == 'XGBscore':
        searchspace = {'x': [3, 4, 5, 6, 7]}
    elif score_func.__name__ == 'Lightscore':
        searchspace = {'x': [2**i-1 for i in range(2,8)]}
    para, score = 1, -2
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(searchspace), direction="maximize")
    study.optimize(score_func, n_trials=n_trials, timeout=timeout,n_jobs=n_jobs)
    print(f"{score_func.__name__} timeout 之前试验次数: {len(study.trials)}")
    if study.best_value > score:
        para = study.best_params['x']
        score = study.best_value
    return para

def getparaRF(today_indexx):
    today_index = today_indexx
    return get_best_param(RFscore, n_trials=5)
def getparaLasso(today_indexx):
    today_index = today_indexx
    return get_best_param(Lassoscore, n_trials=15,n_jobs=3)
def getparaRidge(today_indexx):
    today_index = today_indexx
    return get_best_param(Ridgescore, n_trials=15,n_jobs=3)
def getparaLight(today_indexx):
    today_index = today_indexx
    return get_best_param(Lightscore, n_trials=7)

def getparaXGB(today_indexx):
    today_index = today_indexx
    return get_best_param(XGBscore, n_trials=5)
def getparaNN(today_indexx,time_out=1800):
    today_index = today_indexx
    return get_best_param(NNscore)

start_time = time.time()
Rset=pd.DataFrame(index=[all_date[20:]],columns=['Lasso5sR^2','Lasso30sR^2','Ridge5sR^2','Ridge30sR^2', 'RF5sR^2','RF30sR^2','LGBM5sR^2','LGBM30sR^2'])
Hyperset = pd.DataFrame(index=[all_date[20:len(all_date)-20:20]],columns=['lasso','rf','ridge','light',])
for i in range(20,len(all_date)-20,20):
    print(f"--{i}th cycle of the year--")
    # Get parameter: Lasso->RF->Ridge->Light
    Lassopara=getparaLasso(i-20)
    RFpara=getparaRF(i-20)
    Ridgepara=getparaRidge(i-20)
    Lightpara=getparaLight(i-20)
    Hyperset.loc[all_date[i]]=[Lassopara,RFpara,Ridgepara,Lightpara,]
    
    
    for j in tqdm(range(20)):
        try:
            start_time_sample = time.time()
            x_sample=Xset.loc[all_date[(i+j-5):(i+j)]]
            y_sample1=yset1.loc[all_date[(i+j-5):(i+j)]]
            y_sample2=yset2.loc[all_date[(i+j-5):(i+j)]]
            x_outsample=Xset.loc[all_date[i+j]]
            y_outsample1=yset1.loc[all_date[i+j]]
            y_outsample2=yset2.loc[all_date[i+j]]
            scaler=StandardScaler().fit(x_sample)
            x_sample=scaler.transform(x_sample)
            x_outsample=scaler.transform(x_outsample)
            
            # Lasso
            lasso=Lasso(alpha=Lassopara)
            lasso.fit(x_sample,y_sample1)
            y_predl1=lasso.predict(x_outsample)
            Rset.loc[all_date[i+j],'Lasso5sR^2']=r2_score(y_outsample1,y_predl1)
            lasso.fit(x_sample,y_sample2)
            y_predl2=lasso.predict(x_outsample)
            Rset.loc[all_date[i+j],'Lasso30sR^2']=r2_score(y_outsample2,y_predl2)
            end_time_lasso = time.time()
            
            # 获取被选中的变量
            feature_importance['lasso'].iloc[i+j] = lasso.coef_
            
            # RF
            clf_rf=RandomForestRegressor(n_estimators=50,max_samples=0.01,max_depth=RFpara)
            clf_rf.fit(x_sample,y_sample1)
            y_predrf1=clf_rf.predict(x_outsample)
            Rset.loc[all_date[i+j],'RF5sR^2']=r2_score(y_outsample1,y_predrf1)
            clf_rf.fit(x_sample,y_sample2)
            y_predrf2=clf_rf.predict(x_outsample)
            Rset.loc[all_date[i+j],'RF30sR^2']=r2_score(y_outsample2,y_predrf2)
            
            # 获取特征重要性
            importances = clf_rf.feature_importances_
            feature_importance['rf'].iloc[i+j] = importances
            
            # Ridge
            ridge = Ridge(alpha=Ridgepara)
            ridge.fit(x_sample, y_sample1)
            y_predr1 = ridge.predict(x_outsample)
            Rset.loc[all_date[i+j], 'Ridge5sR^2'] = r2_score(y_outsample1, y_predr1)
            ridge.fit(x_sample, y_sample2)
            y_predr2 = ridge.predict(x_outsample)
            Rset.loc[all_date[i+j], 'Ridge30sR^2'] = r2_score(y_outsample2, y_predr2)

            # Lightgbm
            lgbm = LGBMRegressor(n_estimators=50, subsample=0.5,num_leaves=Lightpara, random_state=42,n_jobs=40)
            lgbm.fit(x_sample, y_sample1)
            y_predlgbm1 = lgbm.predict(x_outsample)
            Rset.loc[all_date[i+j], 'LGBM5sR^2'] = r2_score(y_outsample1, y_predlgbm1)
            lgbm.fit(x_sample, y_sample2)
            y_predlgbm2 = lgbm.predict(x_outsample)
            Rset.loc[all_date[i+j], 'LGBM30sR^2'] = r2_score(y_outsample2, y_predlgbm2)
            
            importances = lgbm.feature_importances_
            feature_importance['lgb'].iloc[i+j] = importances


            new_df.loc[all_date[i+j],new_df.columns[3]]=list(y_predl1)
            new_df.loc[all_date[i+j],new_df.columns[4]]=list(y_predl2)
            new_df.loc[all_date[i+j],new_df.columns[5]]=list(y_predrf1)
            new_df.loc[all_date[i+j],new_df.columns[6]]=list(y_predrf2)

        except Exception as e:
            print(f"An error occurred at {all_date[i+j]}: {e}")
            print(f"Shape of x_outsample: {x_outsample.shape}")

end_time = time.time();total_time = end_time - start_time
print(f"Training/Hyper Parameter Tuning completed! Used {total_time}s for {len(all_date)} trading days of data.")

Rset.to_csv("/data/work/yangsq/Rset_all/%sRset.csv" %stockname)
Hyperset.to_csv("/data/work/yangsq/Hyperset_all/%sHyperset.csv" %stockname)
for name in ['rf','lasso','lgb']:
    feature_importance[name].to_csv("/data/work/yangsq/Feature_importance_all/%s/%sFeature_importance.csv" %(name,stockname))
new_df.to_pickle("/data/work/yangsq/Return_all/%sValue.pickle" %stockname)