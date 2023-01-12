import pandas as pd
import numpy as np
from xgboost import XGBClassifier

def getData(filepath='./data/train.csv'):
    data = pd.read_csv(filepath)
    return data

def preprocess(data):
    Sex = data['Sex']
    Sex = [1 if x == 'female' else 0 for x in Sex]
    Sex = pd.DataFrame({'Sex': Sex})

    cols_to_use = ['Pclass']
    X = pd.concat([data[cols_to_use], Sex], axis=1)

    try:
        y = data.Survived
    except:
        y = -1

    X = np.array(X)
    y = np.array(y)

    return X, y

def fitXGBoost(X, y):
    bst = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.0001, objective='binary:logistic',
                        tree_method='gpu_hist', gpu_id=0)
    bst.fit(X, y)

    return bst

def makeSubmissionFile(model):
    data = pd.read_csv('./data/test.csv')
    X_test, dummies = preprocess(data)
    predictions = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': data.PassengerId, 'Survived': predictions})
    output.to_csv('submission.csv', index=False)
    print("Your submission was successfully saved!")

if __name__=='__main__':
    data = getData()
    X, y = preprocess(data)

    model = fitXGBoost(X, y)

    makeSubmissionFile(model)
