import pandas as pd
import numpy as np
import statsmodels.api as sm

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

def gradient_descent(X, y):
    epochs = 100000
    min_gradient = 0.00001
    learning_rate = 0.001

    m = np.zeros(X.shape[1], dtype=np.float64)
    c, n = 0.0, len(y)

    for epoch in range(epochs):
        m_partial = np.zeros(X.shape[1], dtype=np.float64)
        c_partial = 0.0

        for i in range(n):
            y_pred = (m * X[i]).sum() + c
            m_partial += (y_pred - y[i]) * X[i]
            c_partial += (y_pred - y[i])

        m_partial *= 2/n
        c_partial *= 2/n

        delta_m = -learning_rate * m_partial
        delta_c = -learning_rate * c_partial

        if ((np.abs(delta_m) < min_gradient).all() and abs(delta_c) < min_gradient):
            break

        m = m + delta_m
        c = c + delta_c

        if (epoch % 1000 == 0):
            print('epoch:', epoch, 'delta_m=', delta_m, 'delta_c=', delta_c, 'm=', m, 'c=', c)

    return m, c

if __name__=='__main__':
    X, y = preprocessData()

    least_square(X, y)

    m, c = gradient_descent(X, y)
    print('Results of Gradient Descent are m=', m, 'c=', c)
