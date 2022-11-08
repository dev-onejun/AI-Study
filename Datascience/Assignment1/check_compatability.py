#!/usr/bin/env python
# coding: utf-8

# In[27]:


""" 1번)
첨부된 엑셀 파일 score.xlsx를 입력파일로 받아서, dataframe으로 읽어들인 다음
midterm와 final이 모두 20점 이상인 학생의 학번, 중간고사, 기말고사를
학번 순으로 출력하는 python 프로그램을 작성하라.
"""
import pandas as pd

def sortByStudentNumber(e):
    return e['sno']

result = []

# Read data from excel
excel_file = 'score.xlsx'
data = pd.read_excel(excel_file)

# Filter data, according to the problem conditions
for i in range(len(data)):
    midterm_score = data['midterm'][i]
    final_score = data['final'][i]
    
    if midterm_score >= 20:
        if final_score >= 20:
            result.append(data.iloc[i][['sno', 'midterm', 'final']])

# Sort data, according to the problem conditions
result.sort(key=sortByStudentNumber)

# Print data
result = pd.DataFrame(result)
print(result)


# In[9]:


import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv('HOST')
USER = os.getenv('USER')
PASSWORD = os.getenv('PASSWORD')
DATABASE = os.getenv('DATABASE')


# In[53]:


""" 2번)
1번의 엑셀파일을 mysql 데이터베이스 안에 score 테이블로 생성하라.
"""
import pymysql

connection = pymysql.connect(host=HOST, user=USER, password=PASSWORD, db=DATABASE, cursorclass=pymysql.cursors.DictCursor)
with connection:
    with connection.cursor() as cursor:
        sql = """
            create table if not exists score(
                student_number	int		primary key,
                attendance	decimal(3,2),
                homework	decimal(4,2),
                discussion	int,
                midterm		decimal(4,2),
                final		decimal(4,2),
                score		decimal(4,2),
                grade 		char(1)
            );
            """
        cursor.execute(sql)

        sql = """
            insert IGNORE into score(student_number, attendance, homework, discussion, midterm, final, score, grade)
            values(%s, %s, %s, %s, %s, %s, %s, %s);
        """
        cursor.executemany(sql, list([(x) for x in data.values.tolist()]))
        connection.commit()
        
        cursor.close()


# In[104]:


""" 3번)
score 테이블에 대하여 midterm와 final이 모두 20점 이상인 학생의 학번,중간고사,기말고사를
학번 순으로 출력하는 python 프로그램을 작성하라.
"""

result = pd.DataFrame()

connection = pymysql.connect(
    host=HOST,
    user=USER,
    password=PASSWORD,
    db=DATABASE,
    cursorclass=pymysql.cursors.DictCursor
)

with connection:
    with connection.cursor() as cursor:
        sql = "select student_number, midterm, final from score where midterm >= 20 and final >= 20"
        row = cursor.execute(sql)

        row = cursor.fetchone()
        while row:
            temp = pd.DataFrame([[row['student_number'], row['midterm'], row['final']]],
                                    columns=['student_number', 'midterm', 'final'])
            result = pd.concat([result, temp], ignore_index=True)
            row = cursor.fetchone()
        
        cursor.close()

result.sort_values(by=['student_number'], inplace=True)
print(result)

