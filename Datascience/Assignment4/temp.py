import pandas as pd
import numpy as np
import statsmodels.api as sm
import time

def stopwatch_begin():
    return time.time()

def stopwatch_end(start_time):
    end_time = time.time()
    response_time = end_time - start_time
    print('response time=%f seconds' % (response_time))

def preprocessData():
    EXCEL_FILE = 'db_score.xlsx'
    df = pd.read_excel(EXCEL_FILE)

    X = df[['homework', 'attendance', 'final']]
    X = np.array(X)

    y = df['score']
    y = np.array(y)

    return X, y

def least_square(X, y):
    X = sm.add_constant(X)

    model = sm.OLS(y,X)
    results = model.fit()

    print(results.summary())

    return results.params

def gradient_descent(X, y):
    start_time = stopwatch_begin()

    epochs = 1000000
    min_gradient = 0.000001
    #learning_rate = 0.001
    learning_rate_m = 0.001
    learning_rate_c = 0.1

    m = np.zeros(X.shape[1])
    c, n = 0.0, len(y)

    for epoch in range(epochs):
        y_pred = np.sum(m * X,axis=1) + c
        m_partial = np.sum(2*((y_pred-y) * np.transpose(X)), axis=1) / n
        c_partial = np.sum(2*(y_pred-y)) / n

        delta_m = -learning_rate_m * m_partial
        delta_c = -learning_rate_c * c_partial

        #if ((np.abs(delta_m) < min_gradient).all() and abs(delta_c) < min_gradient):
        if np.all(np.abs(delta_m)) < min_gradient and abs(delta_c) < min_gradient:
            break

        m += delta_m
        c += delta_c

        if (epoch % 1000 == 0):
            print('epoch:', epoch, 'delta_m=', delta_m, 'delta_c=', delta_c, 'm=', m, 'c=', c)

    stopwatch_end(start_time)

    return m, c

if __name__=='__main__':
    X, y = preprocessData()

    ls_params = least_square(X, y)
    m, c = gradient_descent(X, y)

    print('Results of Least Square Method are params=', ls_params)
    print('Results of Gradient Descent are m=', m, 'c=', c)
