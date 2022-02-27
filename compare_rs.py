# IMPORT ------
    #* Dataframe 
import pandas as pd 
    #* Matrices 
import numpy as np
    #* Preprocessing 
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
    #* Splitter 
from sklearn.model_selection import train_test_split

#* Gradient boosting 
from catboost import Pool, CatBoostRegressor, train
    # Metrics 
from sklearn.metrics import mean_absolute_error
    # Misc
import os 

train_mae = []
val_mae = []
test_mae = []

random_state = [1,2,3,4,5,20,30,51,6478,21,56,12,69,84,42,105,1245,24,19,65]

def run_cat_boost(rs:int):
    """_summary_

    Args:
        rs (int): random_state to define for our model

    Returns:
        train, val, test: return MAE scores for each sets
    """
    
    # PATH ------
    # base_path = "/home/jovyan/work"  #"/Maturite_dentaire"
    data_path = "data/Teeth"
    data_name= "dataset.csv"

    # CWD ------
    # os.chdir(base_path)
    
    pandas_types= {'ID':'int','VAL_I1':'category','VAL_I2':'category',
            'VAL_C1':'category','VAL_PM1':'category',
            'VAL_PM2':'category','VAL_M1':'category',
            'VAL_M2':'category','VAL_M3':'category'}

    # ORDINAL ENCODING ------
    ord_encoding= {'A':2,'B':3,'C':4,'D':5,'E':6,'F':7,'G':8,'H':9}
    
    # IMPORT DATA ------
    df = pd.read_csv(os.path.join(data_path,data_name),
                    sep=';',
                    dtype=pandas_types)

    # Remove useless columns ---
    X = df.drop(['ID', 'PAT_AGE'], axis=1)
    Y = df['PAT_AGE']
    
    X_train,X_test, y_train, y_test = train_test_split(X,Y, random_state= rs,train_size=0.8)  
    
    # Simple Ordinal Encoding ---
    X_train.replace(ord_encoding, inplace=True)
    X_test.replace(ord_encoding, inplace=True)
    
    # Remove all samples for which at least 7 out of 8 teeth info is missing --- 
    to_rm = np.where(np.sum(pd.isna(X_train),axis=1)>=8)[0] # rowsums : looking for >= 7
    X_train = X_train.reset_index(drop=True).drop(to_rm,axis=0) # remove rows from training 
    y_train = y_train.reset_index(drop=True).drop(to_rm,axis=0) # remove rows from labels 

    # reset indexes --- 
    X_train.reset_index(drop= True, inplace=True)
    y_train.reset_index(drop= True, inplace=True)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, random_state= rs,test_size=0.1)

    # IMPUTING STRATEGY ------
    knni= KNNImputer(n_neighbors= 40,weights= "distance",add_indicator= True)
    X_train= knni.fit_transform(X_train.values) 
    X_val= knni.transform(X_val.values) 
    X_test= knni.transform(X_test.values) 

    # New columns --- 
    indicator_features= []
    for i in X.columns.values[1:]:
        indicator_features.append(i+ "_missing_indicator")
    new_cols= [*list(X.columns), *indicator_features]
    
    # isolation forest 
    #train ---
    Y_pred_train = IsolationForest(random_state=rs,bootstrap=True,contamination=0.05).fit_predict(X_train) # fit outlier detection # Returns -1 for outliers and 1 for inliers.
    inliers = np.where(Y_pred_train == 1)[0] # which samples to keep 
        #* Keep non outliers
    X_train = X_train[inliers,:]
    y_train = y_train.reset_index(drop=True)[inliers]
    
    model = CatBoostRegressor(iterations=9000, 
                            depth=5, 
                            learning_rate=0.001, 
                            loss_function='MAE',
                            train_dir = "mae",
                            random_seed=rs,
                            grow_policy= "SymmetricTree",
                            l2_leaf_reg= 3,
                            use_best_model=True,
                            od_type="Iter",
                            od_wait=10,
                            verbose=0,
                            colsample_bylevel=0.2,
                            )
    #train the model
    model.fit(X= X_train, y= y_train,
            plot=True,
            eval_set=(X_val,y_val))
    
    #keep our results
    train = mean_absolute_error(model.predict(X_train), y_train)
    val = mean_absolute_error(model.predict(X_val), y_val)
    test = mean_absolute_error(model.predict(X_test), y_test)
    return train, val, test


if __name__ == '__main__':
    # extract our scores
    for i in random_state:
        tr, v, te = run_cat_boost(i)
        train_mae.append(tr)
        val_mae.append(v)
        test_mae.append(te)

    #Store our results in a df
    results = pd.DataFrame({'random_state':random_state, 
                            'train':train_mae, 
                            'val':val_mae, 
                            'test': test_mae})
    
    #export our results to csv
    results.to_csv("results.csv", index=False)