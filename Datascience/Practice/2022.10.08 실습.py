from db_conn import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree

def import_iris_data():
    conn, cur = open_db()

    drop_sql = """drop table if exists iris;"""
    cur.execute(drop_sql)
    conn.commit()

    create_sql = """
        create table iris(
            id int auto_increment primary key,
            sepal_length float,
            sepal_width float,
            petal_length float,
            petal_width float,
            species varchar(20)
            );
    """
    cur.execute(create_sql)
    conn.commit()

    file_name = 'iris.csv'
    iris_data = pd.read_csv(file_name)

    rows = []

    for t in iris_data.values:
        rows.append(tuple(t))

    insert_sql = """
        insert into iris(sepal_length, sepal_width, petal_length, petal_width, species)
            values(%s,%s,%s,%s,%s);
    """

    cur.executemany(insert_sql, rows)
    conn.commit()

    close_db(conn, cur)

def load_iris_data():
    conn, cur = open_db()

    sql = """
        select * from iris;
    """
    cur.execute(sql)

    data = cur.fetchall()   # 데이터 갯수가 몇 천개까지는 메모리에 문제가 생기지 않는다. 그 이상은 fetchone으로 읽고 버리고 해줘야 한다.

    close_db(conn, cur)

    X = [(t['sepal_length'], t['sepal_width'], t['petal_length'], t['petal_width']) for t in data]
    X = np.array(X) # input을 numpy로 통일

    y = [1 if t['species'] == 'Iris-versicolor' else -1 for t in data]
    y = np.array(y) # y를 1차원 list에서 numpy로 바꾸어줌

    return X, y

def iris_classification_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    dtree = tree.DecisionTreeClassifier()
    dtree_model = dtree.fit(X_train, y_train)

    y_predict = dtree_model.predict(X_test)

    acc, prec, rec, f1 = classification_performance_evaluation(y_test, y_predict)

    print('accuracy', acc)
    print('precision', prec)
    print('rec', rec)
    print('f1', f1)

def classification_performance_evaluation(y_test, y_predict):
    tp, tn, fp, fn = 0, 0, 0, 0 # True Positive, True Negative, ...

    for y, yp in zip(y_test, y_predict):    # zip은 2개의 array에 대해 하나씩 읽는 것(이중 for문 필요 x)
        if y == 1 and yp == 1:
            tp += 1
        elif y == 1 and yp == -1:
            fn += 1
        elif y == -1 and yp == 1:
            fp += 1
        else: # y == -1 and yp == -1
            tn += 1
    #positive, negative는 예측을 따라가고, 예측값과 실제의 비교가 True, false다.

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = (tp) / (tp + fp)
    recall = (tp) / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1_score

if __name__=='__main__':
    """import_iris_data()"""
    X, y = load_iris_data()
    iris_classification_train_test(X, y)


