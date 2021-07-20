import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

"""## Classificação"""

# Coleta de Dados
df = pd.read_csv('https://pycourse.s3.amazonaws.com/temperature.csv')

# Define a coluna date como index
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Definição do X e Y
x, y = df[['temperatura']].values, df[['classification']].values
print('x = ', x)
print('y = ', y)

# conversão de Y para valores numericos
le = LabelEncoder()
y = le.fit_transform(y.ravel())
print('y = ', y)

# Classificador
clf = LogisticRegression()
clf.fit(x, y)

# gerando 100 valores de temperaturas
x_test = np.linspace(start = 0., stop = 45., num = 100).reshape(-1,1)

# predição de x_test
y_pred = clf.predict(x_test)

# conversão dos valores y_test pro valor original
y_pred = le.inverse_transform(y_pred)

# saida
saida = {
    'new_temp': x_test.ravel(),
    'new_class': y_pred.ravel()
}
df_saida = pd.DataFrame(saida)

df_saida.info()

df_saida.describe()

# contagem dos valores gerados
df_saida['new_class'].value_counts().plot.bar(
    figsize = (10,7),
    rot = 0,
    title = '# de novos valores gerados'
)

# boxplot
df_saida.boxplot(by = 'new_class', figsize = (10,7))

# Sistema Automatico
def classify_temp():
    ask = True
    while ask:
        temp = float(input('Insira a temperatura'))
        temp = np.array(temp).reshape(-1,1)
        class_temp = clf.predict(temp)
        class_temp = le.inverse_transform(class_temp)
        print(f'A classificação da temperatura {temp.ravel()[0]} é: {class_temp[0]}')
        ask = input('Nova Classificação (y/n): ') == 'y'

# chamando a sistema automatico
classify_temp()

"""## Regressão Linear"""

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

# tranformando para numpy arraye vetor coluna
x, y = np.array(x).reshape(-1,1), np.array(y).reshape(-1,1)

# modelo
reg = LinearRegression()
reg.fit(x, y)

# valor estimado de a e b
print('a = ', reg.coef_.ravel()[0])
print('b = ', reg.intercept_[0])

# predição do modelo
y_pred = reg.predict(x)

# score do modelo
print('score = ', reg.score(x,y))