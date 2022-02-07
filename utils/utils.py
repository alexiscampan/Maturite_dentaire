import numpy as np 
import pandas as pd 
from sklearn.metrics import make_scorer,mean_squared_log_error


# EVALUATE PREDS ------
def evaluate_kaggle(estimator:list, X:np.array, ids= np.array, export:bool=True)->pd.DataFrame:
    # predict
    preds =[model.predict(new_test.values) for model in ensemble]
    # ensemble 
    y_pred = np.array(preds).mean(0) # mean over columns
    
    # to df 
    res=pd.DataFrame(np.transpose(np.array([ids,y_pred])), columns = ["Id","SalePrice"])
    res["Id"]=res["Id"].astype(np.int32)
    if export:
        res.to_csv('submission.csv', index = False)
    return res


# CUSTOM SCORE FUNCTION ------
def rmsle(y_true, y_pred):
    loss=mean_squared_log_error(y_true, y_pred,squared=True) # MSLE
    #loss= np.sqrt(loss) # RMSLE 
    #loss= np.negative(loss) # -RMSLE greater_is_better=False will do it
    return loss

def rmsle_scorer():
    return make_scorer(rmsle,greater_is_better=False)
