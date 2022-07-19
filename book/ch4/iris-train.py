from sklearn import svm, metrics
import random, re

csv = []
with open('iris.csv', 'r', encoding='utf-8') as fp:
    for line in fp:
        line = line.strip()     # 줄바꿈 제거
        cols = line.split(',')  # 쉼표로 자르기

        # 문자열 데이터를 숫자로 변환하기
        fn = lambda n : float(n) if re.match(r'^[0-9\.]+$', n) else n
        cols = list(map(fn, cols))
        csv.append(cols)

# 맨 앞줄의 헤더 제거
del csv[0]
# 데이터 섞기
random.shuffle(csv)

total_len = len(csv)
train_len = int(total_len * 2 / 3)

train_data = []
train_label = []
test_data = []
test_label = []

for i in range(total_len):
    data = csv[i][0:4]
    label = csv[i][4]
    if i < train_len:
        train_data.append(data)
        train_label.append(label)
    else:
        test_data.append(data)
        test_label.append(label)

clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

ac_score = metrics.accuracy_score(test_label, pre)
print('정답률=', ac_score)
