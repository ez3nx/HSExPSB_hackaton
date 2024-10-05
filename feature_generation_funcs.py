import pandas as pd
import numpy as np
from workalendar.europe import Russia
from tqdm import tqdm

num_to_month_mapper = {
    1: 'Jan', 2: 'Feb', 3: 'March',
    4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep',
    10: 'Oct', 11: 'Nov', 12: 'Dec'
}

def data_baseline_prepare_(data):
    data_ = data.copy(deep=True)
    # Дата бронирования
    data_["booking_date_dd_mm_yy"] = pd.to_datetime(
        data_["Дата бронирования"]
    ).dt.strftime("%d-%m-%y")
    data_["booking_date_day"] = data_["Дата бронирования"].dt.day  # пусть будет
    data_["booking_date_month"] = data_["Дата бронирования"].dt.month
    data_["booking_date_month_name"] = data_["booking_date_month"].map(
        num_to_month_mapper
    )
    data_["booking_date_year"] = data_["Дата бронирования"].dt.year
    data_["booking_date_year_month"] = (
        data_["Дата бронирования"].dt.year * 100 + data_["Дата бронирования"].dt.month
    )

    # Заезд
    data_["check_in_date_dd_mm_yy"] = pd.to_datetime(data_["Заезд"]).dt.strftime(
        "%d-%m-%y"
    )
    data_["check_in_delta_booked"] = (
        data_["Заезд"] - data_["Дата бронирования"]
    ).dt.days
    data_["check_in_date_year_month"] = (
        data_["Заезд"].dt.year * 100 + data_["Заезд"].dt.month
    )
    
    # Выезд
    data_["check_out_date_dd_mm_yy"] = pd.to_datetime(data_["Выезд"]).dt.strftime(
        "%d-%m-%y"
    )
    data_["check_out_date_year_month"] = (
        data_["Выезд"].dt.year * 100 + data_["Выезд"].dt.month
    )
    data_["Источник_agg"] = (
        data_["Источник"]
        .apply(
            lambda x: x
            if x
            in ["Официальный сайт", "Программа лояльности", "Бронирование из экстранета"]
            else x.split()[0]
        )
        .astype("category")
    )
    data_["Категория_номера_agg"] = (
        data_["Категория номера"]
        .apply(lambda x: pd.Series(x.split("\n")).mode()[0]
        )
        .astype("category")
    )
    
    data_['non_confirmed'] = data_['Способ оплаты'].str.contains('Отложенная электронная оплата').astype('int')

    print(f"in {data.shape[1]} >>> processing... >>> out {data_.shape[1]}")

    return data_


def calendar_features_(data):
    calendar_rus = Russia()
    data_ = data.copy(deep=True)
    # праздник / рабочий день (заезд)
    tqdm.pandas()
    data_['check_in_is_holiday'] = (
        data_['Заезд'].progress_apply(lambda x: 0 if calendar_rus.is_working_day(x) else 1)
    )
    # праздник / рабочий день (бронь)
    data_['booking_date_is_holiday'] = (
        data_['Дата бронирования'].progress_apply(lambda x: 0 if calendar_rus.is_working_day(x) else 1)
    )
    # праздник / рабочий день (выезд)
    data_['check_out_is_holiday'] = (
        data_['Выезд'].progress_apply(lambda x: 0 if calendar_rus.is_working_day(x) else 1)
    )
    # день недели (бронь + заезд + выезд)
    data_['check_in_weekday'] = data_['Заезд'].dt.weekday
    data_['booking_date_weekday'] = data_['Дата бронирования'].dt.weekday
    data_['check_out_weekday'] = data_['Выезд'].dt.weekday
    
    data_['check_in_week'] = data_['Заезд'].dt.isocalendar().week
    data_['check_in_day'] = data_['Заезд'].dt.dayofyear
    
    print(f"in {data.shape[1]} >>> processing... >>> out {data_.shape[1]}")

    return data_

def sin_cos_coder(data: pd.DataFrame, col_name = ['booking_date_day', 'booking_date_month', 'check_in_weekday', 'check_in_week']):
    """
    Создание sin и cos признаков по дате
    """
    data_ = data.copy(deep=True)
    for col in col_name:
        max_value = data_[col].max()
        data_[f'{col}_sin'] = np.sin(2 * np.pi * data_[col]/max_value)
        data_[f'{col}_cos'] = np.cos(2 * np.pi * data_[col]/max_value)
    return data_

def price_features_(data):
    data_ = data.copy(deep=True)
    
    data_['Цена_за_ночь'] = data_['Стоимость'] / data_['Ночей']
    # в среднем цена по каждой гостнице в будние / выходные и праздники по неделям
    data_["avg_price_weekly"] = data_\
        .groupby(by=["Гостиница", 'Номеров', "Категория номера", "check_in_week", "check_in_is_holiday"])["Цена_за_ночь"] \
        .transform("mean")

    # медианная цена по каждой гостнице в выходные и праздники по неделям
    data_["median_price_weekly"] = data_ \
        .groupby(by=["Гостиница", 'Номеров', "Категория номера", "check_in_week", "check_in_is_holiday"])["Цена_за_ночь"] \
        .transform("median")
    # в среднем цена по каждой гостнице в конкретный день недели каждого месяца
    data_["avg_price_weekday_month"] = data_ \
        .groupby(by=["Гостиница", 'Номеров', "Категория номера", "check_in_weekday", "booking_date_month"])["Цена_за_ночь"] \
        .transform("mean")

    # медианная цена по каждой гостнице в конкретный день недели каждого месяца
    data_["median_price_weekday_month"] = data_ \
        .groupby(by=["Гостиница", 'Номеров', "Категория номера", "check_in_weekday", "booking_date_month"])["Цена_за_ночь"] \
        .transform("median")
    
    data_["avg_price_dayofyear"] = data_ \
        .groupby(by=["Гостиница", 'Номеров', "Категория номера", "Гостиница", "check_in_day"])["Цена_за_ночь"] \
        .transform("mean")
    
    data_['is_beneficial'] = np.where(data_['Цена_за_ночь'] < data_["avg_price_dayofyear"], 1, 0)
    data_['is_beneficial_2'] = np.where(data_['Цена_за_ночь'] < data_["avg_price_weekday_month"], 1, 0)
    data_['is_beneficial_3'] = np.where(data_['Цена_за_ночь'] < data_["avg_price_weekly"], 1, 0)

    return data_