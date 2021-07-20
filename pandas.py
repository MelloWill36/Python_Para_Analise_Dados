import pandas as pd

"""## Leitura de dados"""

# CSV
df_csv = pd.read_csv('https://pycourse.s3.amazonaws.com/temperature.csv')

# Excel
excel_file = pd.ExcelFile('https://pycourse.s3.amazonaws.com/temperature.xlsx')
df_excel = pd.read_excel(excel_file, sheet_name = 'Sheet1')

# 3 primeiras linhas do df
print(df_csv.head(3), '\n')

# 3 ultimas linhas do df
print(df_csv.tail(3), '\n')

# describe do df
print(df_csv.describe(), '\n')

# infos do df
print(df_csv.info(), '\n')

# colunas do df
print(df_csv.columns, '\n')

"""## Indexação"""

df.head()

df['date']

df[['date', 'temperatura']]

df.iloc[0:2]

df.loc[:,'temperatura']

# Converte a coluna Date para datetime
df['date'] = pd.to_datetime(df['date'])
df.info()

# setando um index no df
df = df.set_index('date')

df

# filtros
# seleção dos registros acima de 25 graus
df[df['temperatura'] > 25]

# Registros até 2020-03-01
df[df.index <= '2020-03-01']

# Colouna classification onde a date é menor igual 2020-03-01
df.loc[df.index <= '2020-03-01', ['classification']]

"""## Ordenação"""

df.sort_values(by = ['temperatura'])

df.sort_values(by = ['classification', 'temperatura'])

df.sort_values(by = ['classification', 'temperatura'], ascending = False)

df.sort_index()

"""## Visualização da Dados no Pandas"""

df.plot()

df.plot(
    figsize = (10,5), 
    grid = True,
    style = '-o',
    linewidth = 2,
    color = 'Red',
    legend = True
)

df['classification'].value_counts().plot.bar(figsize = (10,5), rot = 0)

df['classification'].value_counts().plot.pie(figsize = (10,7), shadow = True, autopct = '%1.1f%%')

df.groupby(by = ['classification']).mean()

