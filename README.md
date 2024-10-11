# Regressão Linear

Este notebook realiza uma análise preditiva usando a técnica de Regressão Linear com base no conjunto de dados Auto MPG disponível no Kaggle.

## Dataset

### O dataset pode ser encontrado [aqui](https://www.kaggle.com/datasets/uciml/autompg-dataset?resource=download).

### O dataset contém as seguintes colunas:

- `mpg`: Milhas por galão (valor a ser predito)
- `cylinders`: Número de cilindros do veículo
- `displacement`: Deslocamento do motor
- `horsepower`: Potência do motor (cavalo-vapor)
- `weight`: Peso do veículo
- `acceleration`: Tempo de aceleração de 0 a 60 mph
- `model year`: Ano do modelo do veículo
- `origin`: Local de fabricação do veículo
- `car name`: Nome do veículo

## Bibliotecas Utilizadas

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
%matplotlib inline
```
## Importação dos Dados:
O conjunto de dados foi importado diretamente de um arquivo CSV.

```python
df = pd.read_csv("C:\\Users\\kadek\\Downloads\\Linear Regression - Auto-mpg dataset\\Testando o Modelo de Linear Regression\\auto-mpg.csv", sep=',')
df.head()
```
## Análise Inicial dos Dados:
Foi realizada uma inspeção inicial do dataset, mostrando as primeiras linhas e descrições estatísticas.
```python
df.describe()
df.info()
```
## Conversão de Tipos:
A coluna horsepower foi convertida de string para tipo numérico (int64) após verificar que continha valores não numéricos.
```python
numeric_mask = df['horsepower'].apply(lambda x: str(x).isnumeric())
df = df[numeric_mask]
df['horsepower'] = df['horsepower'].astype('int64')
```
## Visualização dos Dados:
Foram criados diversos gráficos para entender melhor os dados, utilizando o Seaborn:
```python
sns.pairplot(df)
sns.heatmap(df.corr(), annot = True)
sns.jointplot(x='weight', y='mpg', data=df, kind='reg')
sns.barplot(y='displacement', x='cylinders', data=df)
sns.barplot(x=df['model year'].value_counts().index, y= df['model year'].value_counts().values, data=df)
sns.lineplot(x='horsepower', y='mpg', data=df)
```
## Preparação dos Dados:
As variáveis foram divididas em preditoras (X) e o rótulo (Y), sendo então escalonadas usando StandardScaler para melhorar a performance do modelo.
```python
X = df.drop(['mpg', 'car name', 'origin'], axis=1)
Y = df['mpg']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
## Seleção de Hiperparâmetros:
Usamos GridSearchCV para encontrar os melhores hiperparâmetros para o modelo de Regressão Linear.
```python
params = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'n_jobs': [1, 10, 20, 30, 40, 50],
    'positive': [True, False],
}

grid_search = GridSearchCV(LinearRegression(), params, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, Y_train)
grid_search.best_params_
```
## Ajuste e Predição do Modelo:
Com os melhores hiperparâmetros, ajustamos o modelo e realizamos previsões nos dados de teste.
```python
ins = LinearRegression(**grid_search.best_params_)
ins.fit(X_train, Y_train)
pred = ins.predict(X_test)
```
## Avaliação do Modelo:
O desempenho do modelo foi avaliado usando o coeficiente de determinação (R²) e o erro quadrático médio (RMSE).
```python
print('R²:', ins.score(X_test, Y_test))
print('RMSE:', np.sqrt(mean_squared_error(Y_test, pred)))
```
## Também utilizamos cross_val_score para verificar a generalização do modelo:
```python
cross = cross_val_score(ins, X_test, Y_test, cv=10)
final = sum(cross) / len(cross)
final
```
## Teste de Generalização:
### Finalmente, testamos o modelo com novos dados:
```python
X_real = pd.DataFrame([["4","169","79","2277", "18", "82" ]],
                      columns=["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year"])
X_real = sc.transform(X_real)
pred = ins.predict(X_real)
pred
```
### O modelo previu um valor de aproximadamente 23.75 milhas por galão para o novo veículo.

## Conclusão:
A Regressão Linear apresentou uma boa performance para prever o consumo de combustível (mpg) com um coeficiente de determinação de 0.805 e RMSE de 3.09. O modelo foi capaz de generalizar bem nos dados de teste.
