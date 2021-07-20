import numpy as np
import matplotlib.pyplot as plt

# dados
x = [-1, -0.77777778, -0.555555559, -0.33333333, -0.1111111111, 0.11111111111, 0.3333333333, 0.555555555, 0.777777778, 1]
y = [-1.111189389, -0.55859058, -0.2098098, 0.54495884, 0.4993839, 1.1429048, 1.6409090, 2.13434234, 2.6434333, 2.95334342]

# plot dos dados
plt.figure(figsize = (10, 5))
plt.plot(x, y, 'o', label = 'Dados Originais')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

"""
Iremos Estimar uma função do tipo: y = a*x + c
ou seja, devemos achar quais valores de a e b 
que melhor representam os dados.

Os valores reais de a e b são: (a, b) = 2, 1
"""
# tranformando para numpy arraye vetor coluna
x, y = np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)

# adicionando bias: para estimar o termo b
x = np.hstack((x, np.ones(x.shape)))
print(x)
# estimando a e b
beta = np.linalg.pinv(x).dot(y)
print('a estimado:', beta[0][0])
print('b estimado:', beta[1][0])

# plot dos dados
plt.figure(figsize=(10, 5))
plt.plot(x, y, 'o', label='dados originais')
plt.plot(x, x.dot(beta), label='regressao linear')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()