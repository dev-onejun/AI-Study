import pandas as pd
import numpy as np

#from xgboost import XGBClassifier
import tensorflow as tf
#from sklearn.ensemble import RandomForestClassifier
#from supervised.automl import AutoML

def getData(filepath='./data/train.csv'):
    data = pd.read_csv(filepath)
    return data

def preprocess(data):
    Sex = data['Sex']
    Sex = [1 if x == 'female' else 0 for x in Sex]
    Sex = pd.DataFrame({'Sex': Sex})

    """
    import math
    Age = data['Age']
    Age = [ Age.mean() if math.isnan(x) else x for x in Age]
    Age = pd.DataFrame({'Age': Age})
    """

    cols_to_use = ['Pclass', 'SibSp', 'Parch']
    X = pd.concat([data[cols_to_use], Sex], axis=1)
    #X = pd.concat([data[cols_to_use], Age], axis=1)
    #X = pd.get_dummies(X)

    try:
        y = data.Survived
    except: # ignore that test data don't have Survived columns
        y = -1

    X = np.array(X)
    y = np.array(y)

    return X, y

def fitXGBoost(X, y):

    bst = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.0001, objective='binary:logistic',
                        tree_method='gpu_hist', gpu_id=0)
    bst.fit(X, y)

    return bst

def fitMLP(X, y):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam')

    from keras.callbacks import ModelCheckpoint

    check_point = ModelCheckpoint(
        filepath = 'model/model-{epoch:02d}-{loss:.4f}.hdf5',
        monitor='loss',
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [check_point]

    model.fit(X, y, epochs=1000, callbacks=callbacks_list)

def fitRandomForest(X, y):
    forest = RandomForestClassifier()
    forest.fit(X, y)

    return forest

def fitAutoML(X, y):
    model = AutoML()
    model.fit(X, y)

    return model

def makeSubmissionFile(model):
    data = pd.read_csv('./data/test.csv')
    X_test, dummies = preprocess(data)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.build( (4,4) )
    model.load_weights('model/model-985-0.4083.hdf5')

    predictions = model.predict(X_test)
    predictions = [ 1 if x >= 0.5 else 0 for x in predictions]

    output = pd.DataFrame({'PassengerId': data.PassengerId, 'Survived': predictions})
    output.to_csv('submission.csv', index=False)
    print("Your submission was successfully saved!")

if __name__=='__main__':
    data = getData()
    X, y = preprocess(data)

    #model = fitXGBoost(X, y)
    #model = fitMLP(X, y)
    #model = fitRandomForest(X, y)
    #model = fitAutoML(X, y)

    makeSubmissionFile(None)
