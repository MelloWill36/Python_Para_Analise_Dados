import matplotlib.pyplot as plt
import numpy as np
import pandas

x = [-1., -0.77777778, -0.55555556, -0.33333333, -0.11111111,
     0.11111111, 0.33333333, 0.55555556, 0.77777778, 1.]
y = [-1.13956201, -0.57177999, -0.21697033, 0.5425699, 0.49406657,
     1.14972239, 1.64228553, 2.1749824, 2.64773614, 2.95684202]

x, y = np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)
x = np.hstack((x, np.ones(x.shape)))

print(x)

beta = np.linalg.pinv(x).dot(y)
print('a estimado:', beta[0][0])
print('b estimado:', beta[1][0])

plt.figure(figsize=(10, 5))
plt.plot(x, y, 'o', label='dados originais')
plt.plot(x, x.dot(beta), label='regressao linear')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
