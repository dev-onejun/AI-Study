import numpy as np

v1 = np.array([np.log(4), 0, 0, 0, 0, 1/2*np.log(2), 0, 0, 0, np.log(4)])
v2 = np.array([0, 0, 0, 0, 1/2*np.log(2), 0, 0, 1/2*np.log(2), 0, 0])
v3 = np.array([0, 0, 0, np.log(4), 1/2*np.log(2), 0, 0, 0, 0, 0])
v4 = np.array([0, 0, 0, 0, 0, 1/2*np.log(2), np.log(4), 1/2*np.log(2), 0, 0])

v1_v2_cos_sim = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
v1_v3_cos_sim = np.dot(v1, v3)/(np.linalg.norm(v1)*np.linalg.norm(v3))
v1_v4_cos_sim = np.dot(v1, v4)/(np.linalg.norm(v1)*np.linalg.norm(v4))
v2_v3_cos_sim = np.dot(v2, v3)/(np.linalg.norm(v2)*np.linalg.norm(v3))
v2_v4_cos_sim = np.dot(v2, v4)/(np.linalg.norm(v2)*np.linalg.norm(v4))
v3_v4_cos_sim = np.dot(v3, v4)/(np.linalg.norm(v3)*np.linalg.norm(v4))

print(v1_v2_cos_sim, v1_v3_cos_sim, v1_v4_cos_sim, v2_v3_cos_sim, v2_v4_cos_sim, v3_v4_cos_sim, sep='\n')

