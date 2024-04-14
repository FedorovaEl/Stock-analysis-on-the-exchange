import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#from clearml import Task
#task = Task.init(project_name="my project", task_name="my task")

#calls_df, = pd.read_html("https://ru.tradingview.com/chart/?symbol=NYSE%3AIBM/", header=0, parse_dates=["Call Date"])
#print(calls_df)


# Читаем файл и записываем его как DataFrame (структура из pandas)
df = pd.read_csv('intraday_5min_IBM.csv')

# Задаем на каких столбцах будем учить модель для этого выкидываем столбец open (будем пытаться его предсказать) и timestamp (его модель не может обработать)
X = df.drop(['timestamp', 'open'], axis=1)

# Задаем какой столбец надо предсказать
y = df['open']

# С помощью встроенной функции sklearn распределяем данные на учебные и проверочные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Задаем какого типа будет модель (регрессионная)
model = GradientBoostingRegressor()

#Закидываем ранее подготовленные данные для обучения
model.fit(X_train, y_train)

#print("X_test: " + X_test)
#print("y_test: " + y_test)

# Используем обученную модель для предсказания
predictions = model.predict(X_test)
print(predictions)

#for i in predictions:
 #   print(predictions)
#print("predictions" + predictions)
#print(predictions[0][0])

#Оцениваем погрешность путем сравнения того, что напредсказывали и того что заложили как проверочные данные
mse = mean_squared_error(predictions, y_test)
print("Mean squared error:", mse)