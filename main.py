import sys  # Системные параметры и функции
from datetime import datetime, timedelta  # Работа с датами и временными промежутками
from io import StringIO  # Введение/выведение данных в виде строк

import numpy as np  # Научные вычисления и операции с массивами
import optuna  # Библиотека для оптимизации гиперпараметров
import pandas as pd  # Работа с данными в табличной форме
import requests  # Отправка HTTP-запросов
from PIL._imaging import display
from catboost import CatBoostRegressor  # Модель CatBoost для регрессии
from loguru import logger  # Логирование
from matplotlib import pyplot as plt  # Построение графиков
from plotly import express as px  # Создание интерактивных графиков
from plotly import graph_objects as go  # Ещё возможности для интерактивных графиков
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Метрики для оценки моделей
from sklearn.model_selection import TimeSeriesSplit, train_test_split  # Методы для разбиения данных на тренировочные и тестовые выборки

def setup_logging(log_file=None):
    """
    Настраивает логирование с помощью Loguru.

    Args:
        log_file (str, optional): Путь к файлу конфигурации.
    """
    logger.remove()

    console_format = (
        "<dim>{time:YYYY-MM-DD HH:mm:ss.SSS}</dim> | "
        "<level>{level:.1s}</level> | "
        "<cyan>{file}</cyan>:<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "\n{message}"
    )
    logger.add(
        sys.stdout, format=console_format, level="DEBUG", colorize=True, enqueue=True
    )

    if log_file:
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level:.1s} | "
            "{file}:{name}:{function}:{line} | "
            "{message}"
        )
        logger.add(
            log_file, rotation="10 MB", level="INFO", format=file_format, enqueue=True
        )

    return logger


logger = setup_logging()

# Нам нужны данные за последние четыре месяца.
# В течение нескольких месяцев данные не поступают, но можно запросить их отдельно по месяцам и объединить в dataframe.

class StockDataLoader:
    """
    Класс для загрузки и подготовки данных о ценах акций с использованием Alpha Vantage API.
    """

    def __init__(self, api_key, base_url="https://www.alphavantage.co/query"):
        """
        Инициализация StockDataLoader.

        Args:
            api_key (str): API ключ для доступа к Alpha Vantage.
            base_url (str, optional): Базовый URL для API запросов. По умолчанию "https://www.alphavantage.co/query".
        """
        self.api_key = api_key
        self.base_url = base_url

    def _load_monthly_data(self, month, year):
        """
        Загрузка данных о ценах акций за конкретный месяц с использованием Alpha Vantage API.

        Args:
            month (int): Месяц (1-12).
            year (int): Год.

        Returns:
            pd.DataFrame: DataFrame с данными о ценах акций или пустой DataFrame в случае ошибки.
        """
        month_str = f"0{month}" if month < 10 else month
        url = f"{self.base_url}?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=60min&apikey={self.api_key}&datatype=csv&month={year}-{month_str}&outputsize=full"
        logger.info(f"Preparing to download data for {month}/{year} from {url}")

        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = pd.read_csv(url)
                logger.info(f"Successfully downloaded data for {month}/{year}")
                return data
            except pd.errors.ParserError:
                logger.error(f"Error parsing data from {url}")
                return pd.DataFrame()
        else:
            logger.error(f"API request failed with status code {response.status_code} for {month}/{year}")
            return pd.DataFrame()

    def load_and_prepare_data(self, months=5):
        """
        Загрузка и подготовка данных за последние несколько месяцев.

        Args:
            months (int, optional): Количество месяцев для загрузки данных. По умолчанию 5.

        Returns:
            pd.DataFrame: Подготовленный DataFrame с данными о ценах акций.
        """
        current_date = pd.to_datetime("today")
        data = pd.DataFrame()

        start_month = current_date.month - months + 1
        start_year = current_date.year if start_month > 0 else current_date.year - 1
        logger.info(
            f"Starting data download from {start_month}/{start_year} to {current_date.month}/{current_date.year}")

        for i in range(months):
            month = (current_date.month - i - 1) % 12 + 1
            year = current_date.year if (current_date.month - i - 1) >= 0 else current_date.year - 1
            logger.info(f"Downloading data for month {month}/{year}")

            monthly_data = self._load_monthly_data(month, year)
            data = pd.concat([data, monthly_data], ignore_index=True) if not monthly_data.empty else monthly_data

        if data.empty:
            logger.warning("No data downloaded for the specified period.")
            return data

        # Очистка данных и генерация признаков
        data.drop_duplicates(inplace=True)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour_of_day'] = data['timestamp'].dt.hour
        logger.info("Data preparation complete, including cleaning and feature engineering.")
        return data


def load_preloaded_data(data_file):
    """
    Загрузка и подготовка данных о ценах акций из CSV файла.

    Args:
        data_file (str): Путь к CSV файлу с данными о ценах акций.

    Returns:
        pd.DataFrame: Подготовленный DataFrame с данными о ценах акций.
    """
    try:
        data = (
            pd.read_csv(data_file)
            .reset_index(drop=True)
            .drop(columns=["Unnamed: 0", "Unnamed: 0.1"])
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {data_file}")

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["hour_of_day"] = data["timestamp"].dt.hour
    data.sort_values("timestamp", inplace=True)
    data = data.reset_index(drop=True)

    return data

def get_data(load_from_file=True, data_file="stock_data.csv", api_key="YOUR_API_KEY", months=5):
    """
    Получение данных о ценах акций путем загрузки из файла или скачивания через API.

    Args:
        load_from_file (bool, optional): Флаг для загрузки данных из файла. Если False, данные будут скачаны через API. По умолчанию True.
        data_file (str, optional): Путь к файлу с данными. По умолчанию "stock_data.csv".
        api_key (str, optional): API ключ для доступа к Alpha Vantage. По умолчанию "YOUR_API_KEY".
        months (int, optional): Количество месяцев для загрузки данных через API. По умолчанию 5.

    Returns:
        pd.DataFrame: Подготовленные данные о ценах акций.
    """
    if load_from_file:
        logger.info(f"Loading data from file: {data_file}")
        return load_preloaded_data(data_file)
    else:
        logger.info(f"Downloading data using API key: {api_key}")
        stock_data_loader = StockDataLoader(api_key)
        return stock_data_loader.load_and_prepare_data(months)


data = get_data(load_from_file=True, data_file="stock_data.csv", api_key="9PJCRGKTVAQXTFWH", months=5)
display(data)

# pd.DataFrame.to_csv(data, "stock_data_final.csv")

def show_unique_hours_count(data):
    unique_hours_count = data['timestamp'].nunique()
    print(
        f"{unique_hours_count} часа/ов, {round(unique_hours_count / 24, 2)} ({int(unique_hours_count / 24)} целых дня/ей)")


show_unique_hours_count(data)


def plot_hourly_dynamics(data):
    # Группировка данных по часу и расчёт средних значений цен закрытия
    hourly_data = data.groupby('hour_of_day').agg({'close': 'mean'}).reset_index()

    # Создание графика
    fig = px.line(hourly_data, x='hour_of_day', y='close', title='Средняя цена закрытия акций по часам',
                  labels={'close': 'Средняя цена закрытия', 'hour_of_day': 'Час дня'},
                  markers=True)
    fig.update_layout(xaxis_title='Час дня',
                      yaxis_title='Средняя цена закрытия',
                      template='plotly_white')
    fig.show()


plot_hourly_dynamics(data)


def plot_hourly_detailed(data):
    df = data.copy()
    df.set_index('timestamp', inplace=True)

    # 1. Скользящие средние
    df['SMA_7'] = df['close'].rolling(window=7).mean()
    df['SMA_30'] = df['close'].rolling(window=30).mean()

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=df.index, y=df['close'], name='Цена закрытия'))
    fig_ma.add_trace(go.Scatter(x=df.index, y=df['SMA_7'], name='7-дневное SMA'))
    fig_ma.add_trace(go.Scatter(x=df.index, y=df['SMA_30'], name='30-дневное SMA'))
    fig_ma.update_layout(title='Скользящие средние и цены закрытия', xaxis_title='Дата', yaxis_title='Цена',
                         template='plotly_white')
    fig_ma.show()

    # 2. Волатильность
    df['Volatility'] = df['close'].rolling(window=30).std()

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=df.index, y=df['Volatility'], name='Волатильность'))
    fig_vol.update_layout(title='Волатильность цен закрытия', xaxis_title='Дата', yaxis_title='Волатильность',
                          template='plotly_white')
    fig_vol.show()

    # 3. Объемы торгов
    fig_vol_trades = go.Figure()
    fig_vol_trades.add_trace(go.Bar(x=df.index, y=df['volume'], name='Объемы торгов'))
    fig_vol_trades.update_layout(title='Объемы торгов', xaxis_title='Дата', yaxis_title='Объем',
                                 template='plotly_white')
    fig_vol_trades.show()


plot_hourly_detailed(data)

# Расширим функцию feature_engineering для добавления новых признаков:
# День недели и день месяца – это может помочь уловить недельные и месячные паттерны в данных.
# Лаговые признаки – предыдущие значения закрытия (close). Например, значение закрытия на один час назад.
# Скользящие средние – например, скользящее среднее закрытия за последние 3 часа.

int(len(data) * 0.8)

int(len(data) * 0.2) // 24

import streamlit as st


class StockPricePredictor:
    """
    Класс для обучения и прогнозирования цен акций с помощью CatBoost.
    """

    def __init__(self, data, target_col='close', test_size=0.2, model_params=None):
        """
        Инициализация класса StockPricePredictor.
        """
        self.data = data
        self.target_col = target_col
        self.test_size = test_size
        self.model_params = model_params or {'iterations': 100, 'learning_rate': 0.1, 'depth': 3}
        self.model = CatBoostRegressor(**self.model_params)
        self.prepare_and_split_data()

    def prepare_and_split_data(self):
        """
        Генерация признаков и разделение данных на обучающую и тестовую выборки.
        """
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data.sort_values(by='timestamp', inplace=True)

        # Генерация признаков
        full_data = self.data.copy()
        full_data = generate_base_features(full_data)

        # Разделение данных на тренировочные и тестовые без утечки данных
        train_size = int(len(full_data) * (1 - self.test_size))
        train_data = full_data.iloc[:train_size].dropna()  # Убираем строки сy NaN значениями
        test_data = full_data.iloc[train_size:].dropna()  # Убираем строки с NaN значениями

        self.X_train = train_data.drop(columns=[self.target_col, 'timestamp'])
        self.y_train = train_data[self.target_col]
        self.X_test = test_data.drop(columns=[self.target_col, 'timestamp'])
        self.y_test = test_data[self.target_col]

        logger.info("Данные успешно подготовлены и разделены на тренировочную и тестовую выборки.")

    def train(self):
        """
        Обучение модели CatBoostRegressor.
        """
        self.model.fit(self.X_train, self.y_train, silent=True)
        logger.info("Модель успешно обучена.")

    def predict(self, X):
        """
        Предсказание цен акций на основе входных данных X.
        """
        return self.model.predict(X)

    def evaluate(self):
        """
        Оценка качества модели на тестовых данных.
        """
        y_pred = self.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {'MSE': mse, 'MAE': mae, 'R2': r2}

    def plot_results(self, title="Фактические против прогнозируемых цен"):
        """
        Построение графика фактических и прогнозируемых цен.
        """
        train_predictions = self.predict(self.X_train)
        test_predictions = self.predict(self.X_test)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.data['timestamp'][:len(self.X_train)],
            y=self.y_train,
            mode='lines',
            name='Фактические цены',
            line=dict(color='blue', width=2)  # Сделаем линии тоньше
        ))

        # Добавление линий между фактическими и предсказанными значениями
        for i in range(1, len(self.X_test)):
            fig.add_trace(go.Scatter(
                x=[self.data['timestamp'].iloc[len(self.X_train) + i - 1],
                   self.data['timestamp'].iloc[len(self.X_train) + i]],
                y=[self.y_test.iloc[i - 1], test_predictions[i]],
                mode='lines',
                line=dict(color='red', width=2),  # Сделаем линии тоньше
                showlegend=False  # Скрыть из легенды
            ))

        # Добавление прогнозов на неделю вперед
        future_predictions = self.predict_week_ahead()
        fig.add_trace(go.Scatter(
            x=future_predictions['timestamp'],
            y=future_predictions['predicted_close'],
            mode='lines',
            name='Прогнозы на неделю вперед',
            line=dict(color='orange', dash='dash')
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Время',
            yaxis_title='Цена',
            legend_title='Легенда',
            template='plotly_white'
        )

        fig.show()

    def plot_zoomed_results(self, title="Фактические против прогнозируемых цен (увеличенный участок)"):
        """
        Построение графика фактических и прогнозируемых цен для последних нескольких недель.
        """
        zoom_weeks = 4  # Количество недель для увеличенного участка
        zoom_data = self.data[self.data['timestamp'] >= self.data['timestamp'].max() - pd.Timedelta(weeks=zoom_weeks)]
        zoom_y_test = self.y_test.iloc[-len(zoom_data):]
        zoom_predictions = self.predict(self.X_test.iloc[-len(zoom_data):])

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=zoom_data['timestamp'],
            y=zoom_y_test,
            mode='lines',
            name='Фактические цены',
            line=dict(color='blue', width=2)  # Сделаем линии тоньше
        ))

        # Добавление линий между фактическими и предсказанными значениями
        for i in range(1, len(zoom_y_test)):
            fig.add_trace(go.Scatter(
                x=[zoom_data['timestamp'].iloc[i - 1], zoom_data['timestamp'].iloc[i]],
                y=[zoom_y_test.iloc[i - 1], zoom_predictions[i]],
                mode='lines',
                line=dict(color='red', width=2),  # Сделаем линии тоньше
                showlegend=False  # Скрыть из легенды
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Время',
            yaxis_title='Цена',
            legend_title='Легенда',
            template='plotly_white'
        )

        fig.show()

    def predict_week_ahead(self):
        """
        Прогноз на неделю вперед.
        """
        last_date = self.data['timestamp'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        future_data = pd.DataFrame({'timestamp': future_dates})

        # Добавляем значения 'close' для последней строки
        future_data['close'] = np.nan
        future_data = pd.concat([self.data, future_data], ignore_index=True)
        future_data = generate_base_features(future_data)

        # Удаляем строки с NaN
        future_data = future_data.dropna().reset_index(drop=True)

        future_data = future_data.iloc[-7:]
        future_features = future_data.drop(columns=['close', 'timestamp'])
        future_predictions = self.predict(future_features)

        future_df = pd.DataFrame({'timestamp': future_dates, 'predicted_close': future_predictions})
        logger.info("Прогнозы на неделю вперед успешно сгенерированы.")
        return future_df


def generate_base_features(data):
    """
    Генерация базовых признаков.
    """
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['day_of_month'] = data['timestamp'].dt.day
    data['lag_close_1'] = data['close'].shift(1)
    data['rolling_mean_7'] = data['close'].shift(1).rolling(window=7).mean()  # Сдвигаем на 1 для предотвращения утечки
    data['rolling_std_7'] = data['close'].shift(1).rolling(window=7).std()  # Сдвигаем на 1 для предотвращения утечки
    data['rolling_mean_30'] = data['close'].shift(1).rolling(
        window=30).mean()  # Сдвигаем на 1 для предотвращения утечки
    data['rolling_std_30'] = data['close'].shift(1).rolling(window=30).std()  # Сдвигаем на 1 для предотвращения утечки
    return data


def objective(trial, data):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 4, 6),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 128),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
        'random_strength': trial.suggest_float('random_strength', 0.5, 5)
    }

    if params['bootstrap_type'] == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
    elif params['bootstrap_type'] == 'Bernoulli':
        params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)

    model = CatBoostRegressor(**params, silent=True)
    full_data = generate_base_features(data)
    train_size = int(len(full_data) * 0.8)
    train_data = full_data.iloc[:train_size]
    test_data = full_data.iloc[train_size:]

    X_train = train_data.drop(columns=['close', 'timestamp'])
    y_train = train_data['close']
    X_test = test_data.drop(columns=['close', 'timestamp'])
    y_test = test_data['close']

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


def optimize_hyperparameters(data, n_trials):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, data), n_trials=n_trials)
    return study.best_params


def evaluate_model(predictor):
    """
    Оценка модели.

    Args:
        predictor (StockPricePredictor): Экземпляр класса StockPricePredictor.

    Returns:
        dict: Результаты оценки модели.
    """
    return predictor.evaluate()

# Оптимизация гиперпараметров
best_params = optimize_hyperparameters(data, n_trials=500)
best_params

# Инициализация и обучение модели
predictor = StockPricePredictor(data, model_params=best_params)
# predictor = StockPricePredictor(data)
predictor.train()

# Оценка модели
evaluation_results = evaluate_model(predictor)
print(evaluation_results)

# Визуализация результатов
predictor.plot_results()

# Визуализация увеличенного участка
predictor.plot_zoomed_results()
