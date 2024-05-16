import sys
import requests
from datetime import datetime
from io import StringIO
from IPython.display import display

from loguru import logger

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

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


class StockDataLoader:
    def __init__(self, api_key, base_url="https://www.alphavantage.co/query"):
        self.api_key = api_key
        self.base_url = base_url

    def _load_monthly_data(self, month, year):
        """
        Loads stock data for a specific month using Alpha Vantage API.

        Args:
          month (int): Month (1-12).
          year (int): Year.

        Returns:
          pd.DataFrame: DataFrame containing stock data or an empty DataFrame on error.
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
        Loads and prepares data for the past months.

        Args:
          months (int, optional): Number of months of data to load. Defaults to 4.

        Returns:
          pd.DataFrame: Prepared DataFrame containing the stock data.
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
            display(monthly_data)
            data = pd.concat([data, monthly_data], ignore_index=True) if not monthly_data.empty else monthly_data

        if data.empty:
            logger.warning("No data downloaded for the specified period.")
            return data

        # Data cleaning and feature engineering
        data.drop_duplicates(inplace=True)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour_of_day'] = data['timestamp'].dt.hour
        logger.info("Data preparation complete, including cleaning and feature engineering.")
        return data

api_key = "9PJCRGKTVAQXTFWH"
stock_data_loader = StockDataLoader(api_key)
data = stock_data_loader.load_and_prepare_data()

data

def load_preloaded_data(data_file):
    """
    Loads and prepares stock data from a CSV file.

    Args:
      data_file (str, optional): Path to the CSV file containing stock data. Defaults to "stock_data.csv".

    Returns:
      pd.DataFrame: Prepared DataFrame containing the stock data.
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


data = load_preloaded_data("stock_data.csv")
data


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



# Загрузка данных
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
    fig_ma.update_layout(title='Скользящие средние и цены закрытия', xaxis_title='Дата', yaxis_title='Цена', template='plotly_white')
    fig_ma.show()

    # 2. Волатильность
    df['Volatility'] = df['close'].rolling(window=30).std()

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=df.index, y=df['Volatility'], name='Волатильность'))
    fig_vol.update_layout(title='Волатильность цен закрытия', xaxis_title='Дата', yaxis_title='Волатильность', template='plotly_white')
    fig_vol.show()

    # 3. Объемы торгов
    fig_vol_trades = go.Figure()
    fig_vol_trades.add_trace(go.Bar(x=df.index, y=df['volume'], name='Объемы торгов'))
    fig_vol_trades.update_layout(title='Объемы торгов', xaxis_title='Дата', yaxis_title='Объем', template='plotly_white')
    fig_vol_trades.show()


plot_hourly_detailed(data)


int(len(data) * 0.8)

int(len(data) * 0.2) // 24


class StockPricePredictor:
    """
    Класс для обучения и прогнозирования цен акций с помощью CatBoost.

    Args:
        data (pd.DataFrame): Исходные данные о ценах акций.
        target_col (str, optional): Название столбца с целевой переменной (цена закрытия). По умолчанию 'close'.
        test_size (float, optional): Доля данных, используемая для тестирования. По умолчанию 0.2.
        model_params (dict, optional): Параметры CatBoostRegressor.
                                       По умолчанию {'iterations': 100, 'learning_rate': 0.1, 'depth': 3}.
    """

    def __init__(self, data, target_col='close', test_size=0.2, model_params=None):
        """
        Инициализация класса StockPricePredictor.

        Args:
            data (pd.DataFrame): Исходные данные о ценах акций.
            target_col (str, optional): Название столбца с целевой переменной (цена закрытия). По умолчанию 'close'.
            test_size (float, optional): Доля данных, используемая для тестирования. По умолчанию 0.2.
            model_params (dict, optional): Параметры CatBoostRegressor.
                                       По умолчанию {'iterations': 100, 'learning_rate': 0.1, 'depth': 3}.
        """
        self.data = data
        self.target_col = target_col
        self.test_size = test_size
        self.model_params = model_params or {'iterations': 100, 'learning_rate': 0.1, 'depth': 3}
        self.model = CatBoostRegressor(**self.model_params)

    def prepare_and_split_data(self):
        """
        Генерация признаков и разделение данных на обучающую и тестовую выборки.
        """
        # Генерация всех признаков для всего датасета
        full_features = generate_base_features(self.data)

        # Объединение исходных данных с новыми признаками
        full_data = pd.concat([self.data, full_features], axis=1)
        display(full_data)

        # Разделение данных
        train_size = int(len(full_data) * (1 - self.test_size))
        train_data = full_data[:train_size]
        test_data = full_data[train_size:]

        feature_cols = [col for col in train_data.columns if col not in [self.target_col, 'timestamp']]

        self.X_train = train_data[feature_cols]
        self.y_train = train_data[self.target_col]
        self.X_test = test_data[feature_cols]
        self.y_test = test_data[self.target_col]

    def train(self):
        """
        Обучение модели CatBoostRegressor.
        """
        self.model.fit(self.X_train, self.y_train, silent=True)

    def predict(self, X):
        """
        Предсказание цен акций на основе входных данных X.

        Args:
            X (pd.DataFrame): Данные для прогнозирования.

        Returns:
            numpy.ndarray: Массив прогнозируемых цен.
        """
        return self.model.predict(X)

    def evaluate(self):
        """
        Оценка качества модели на тестовых данных.

        Returns:
            dict: Словарь с метриками MSE, MAE и R^2.
        """
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {'MSE': mse, 'MAE': mae, 'R2': r2}

    def plot_results(self, title="Фактические против прогнозируемых цен"):
        """
        Строит график фактических и прогнозируемых цен.

        Args:
            title (str, optional): Заголовок графика. По умолчанию "Фактические против прогнозируемых цен".
        """
        # Предсказания на тренировочных данных
        train_predictions = self.predict(self.X_train)

        # Предсказания на тестовых данных
        test_predictions = self.predict(self.X_test)

        # График
        fig = go.Figure()

        # Фактические данные
        fig.add_trace(go.Scatter(
            x=self.data['timestamp'][:len(self.X_train)],
            y=self.y_train,
            mode='lines',
            name='Фактические цены',
            line=dict(color='blue')
        ))

        # Предсказания на тренировочных данных
        fig.add_trace(go.Scatter(
            x=self.data['timestamp'][:len(self.X_train)],
            y=train_predictions,
            mode='lines',
            name='Предсказания на тренировочных данных',
            line=dict(color='green')
        ))

        # Предсказания на тестовых данных
        fig.add_trace(go.Scatter(
            x=self.data['timestamp'][-len(self.X_test):],
            y=test_predictions,
            mode='lines',
            name='Предсказания на тестовых данных',
            line=dict(color='red')
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Время',
            yaxis_title='Цена',
            legend_title='Легенда',
            template='plotly_white'
        )

        fig.show()


def generate_base_features(data):
    """
    Функция для генерации базовых признаков.
    Создает словарь с новыми признаками:
        day_of_week: День недели из timestamp.
        day_of_month: День месяца из timestamp.
        lag_close_1: Лаговая цена закрытия (сдвинутая на 1 период).
    Заполняет пропуски в lag_close_1 методом bfill (заполнение назад).

    Args:
        data (pd.DataFrame): Исходные данные.

    Returns:
        pd.DataFrame: DataFrame, созданный из словаря признаков..
    """
    # Создаем словарь для новых признаков
    features = {}
    features['day_of_week'] = data['timestamp'].dt.dayofweek
    features['day_of_month'] = data['timestamp'].dt.day
    features['lag_close_1'] = data['close'].shift(1).fillna(method='bfill')

    # Возвращаем DataFrame, созданный из словаря признаков
    return pd.DataFrame(features, index=data.index)

# Инициализируем и запускаем наш пайплайн
predictor = StockPricePredictor(data)

# Подготавливаем и разбиваем данные на обучающую и тестовую выборки
predictor.prepare_and_split_data()

# Обучаем модель CatBoostRegressor на обучающих данных
predictor.train()

# Оцениваем качество модели на тестовых данных
evaluation_results = predictor.evaluate()

# Выводим результаты оценки
print(evaluation_results)

predictor.plot_results()
