import pandas as pd
import numpy as np

xl_file = 'db_score.xlsx'
df = pd.read_excel(xl_file)

midterm = df['midterm']
final = df['final']

mean_midterm = midterm.mean()
mean_final = final.mean()

print('mean_midterm = %d' % mean_midterm)
print('mean_final = %d' % mean_final)

cov1 = ((midterm - mean_midterm) * (final - mean_final)).mean()
cov2 = np.cov(midterm, final)
print('cov1', cov1)
print('cov2', cov2)

std_midterm = midterm.std()
std_final = final.std()

correlation1 = cov1 / (std_midterm*std_final)
print(correlation1)

correlation2 = np.corrcoef(midterm, final)
print(correlation2)
