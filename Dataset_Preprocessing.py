"""Metadata_preprocessing.ipynb

# **Изучение данных**

## Импорт библиотек
"""

# импортируем pandas - основную библиотеку аналитика данных
import pandas as pd
# импортируем библиотеку numpy для возможных расчетов
import numpy as np
# импортируем модуль json для распаковки структур данных формата json
import json
# импортируем статистические библиотеки
import scipy
import scipy.stats as stats
# импортируем модуль pyplot из библиотеки matplotlib для визуализации
from matplotlib import pyplot as plt
# импортируем библиотеку seaborn для визуализации
import seaborn as sns
# импортируем регулярные выражения
import re
from sklearn.metrics.pairwise import cosine_similarity
from model_loader import model
from Text_Preprocessing import Text_Preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
from hdbscan import HDBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tqdm import tqdm
import xgboost as xgb
from sklearn.utils.multiclass import unique_labels
from xgboost_utils import load_XGBoost
from rapidfuzz import fuzz, process
from transliterate import translit
from sklearn.cluster import DBSCAN
from umap.umap_ import UMAP
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import pickle
import joblib
import os
from joblib import load
from catboost import Pool
from NER import NER_Preprocessing
import warnings
warnings.simplefilter('error', RuntimeWarning)

class Dataset_Preprocessing():

    def __init__(self):
        pass

    """##Чтение файла"""

    def load_dataset(self):

        with open("ml_ozon_сounterfeit_train.csv", encoding="utf-8") as f:
            df_ozon_train = pd.read_csv(f)

        print(df_ozon_train.head())

        with open("ml_ozon_сounterfeit_test.csv", encoding="utf-8") as f:
            df_ozon_test = pd.read_csv(f)

        print(df_ozon_test.head())

        return df_ozon_train, df_ozon_test


    """##Обзор датасета
    
    Если артикул — уникальный для каждой позиции, его лучше удалить (будет переобучение). Но если он повторяется (например, артикул партии или модели), можно оставить.
    
    1. Категориальные признаки
    категория, бренд, страна производитель, магазин, артикул
    Что делать:
    Оставить как строки. CatBoost умеет работать с категориальными признаками напрямую, без необходимости one-hot encoding.
    
    Просто укажи список категориальных признаков при обучении модели:
    cat_features = ['категория', 'бренд', 'страна производитель', 'магазин']
    2. Числовые признаки
    Поля: цена, рейтинг, количество отзывов
    
    Что делать:
    Привести к числовому типу (float, int).
    
    Заполнить пропущенные значения (если есть), например:
    
    df['рейтинг'] = df['рейтинг'].fillna(df['рейтинг'].median())
    Можно попробовать логарифмирование для цена и количество отзывов, чтобы справиться с перекосом:
    
    df['log_цена'] = np.log1p(df['цена'])
    df['log_отзывы'] = np.log1p(df['количество отзывов'])
    3. Текстовые признаки
    Поля: заголовок, описание
    
    Что делать:
    Используешь FastText для генерации эмбеддингов.
    
    Например, усредни вектор слов FastText по каждому описанию или заголовку:
    
    import fasttext
    model = fasttext.load_model("cc.ru.300.bin")  # или твоя модель
    
    def get_text_vector(text):
       return model.get_sentence_vector(text)
    
    df['desc_vec'] = df['описание'].apply(get_text_vector)
    Важно: если используешь Pandas DataFrame, разобрать desc_vec в отдельные колонки:
    
    desc_vecs = df['desc_vec'].apply(pd.Series)
    desc_vecs.columns = [f'desc_ft_{i}' for i in range(desc_vecs.shape[1])]
    df = pd.concat([df, desc_vecs], axis=1)
    df.drop(columns=['desc_vec'], inplace=True)
    
    Как собрать финальный датасет для CatBoost:
    Категориальные поля: оставить как строки, передать в cat_features.
    
    Числовые поля: обработать, при необходимости логарифмировать.
    
    Тексты: преобразовать через FastText → вектора → добавить как числовые признаки.
    
    Убедись, что все признаки — либо строки (категории), либо float.
    """

    def preprocess_dataset(self, df_products_feedback_questions):

        print(df_products_feedback_questions.head())

        print(df_products_feedback_questions.info())
        """
        RangeIndex: 197198 entries, 0 to 197197
         0   id                            197198 non-null  int64  (Случайно сгенерированный id товара)
         1   resolution                    197198 non-null  int64  (Разметка от операционистов, является ли товар контрафактным или нет)
         2   brand_name                    116667 non-null  object --NaN (Брэнд товара)
         3   description                   171138 non-null  object --NaN (Текстовое описание товара из карточки товара)
         4   name_rus                      197198 non-null  object (Имя в карточке товара)
         5   CommercialTypeName4           197198 non-null  object  (Конкретизированная коммерческая категория товара)
         6   rating_1_count                47193 non-null   float64 --NaN (количество рейтинга 1)
         7   rating_2_count                47193 non-null   float64 --NaN (количество рейтинга 2)
         8   rating_3_count                47193 non-null   float64 --NaN (количество рейтинга 3)
         9   rating_4_count                47193 non-null   float64 --NaN (количество рейтинга 4)
         10  rating_5_count                47193 non-null   float64 --NaN (количество рейтинга 5)
         11  comments_published_count      47193 non-null   float64 --NaN (количество опубликованных комментариев)
         12  photos_published_count        47193 non-null   float64 --NaN (количество опубликованных фотографий)
         13  videos_published_count        47193 non-null   float64 --NaN (количество опубликованных видео)
         14  PriceDiscounted               197198 non-null  float64 (цена со скидкой)
         15  item_time_alive               197198 non-null  int64 (Сколько времени товар находится в продаже до времени проверки)
         16  item_count_fake_returns7      197198 non-null  int64 (количество фиктивных возвратов за неделю)
         17  item_count_fake_returns30     197198 non-null  int64 (количество фиктивных возвратов за месяц)  
         18  item_count_fake_returns90     197198 non-null  int64 (количество фиктивных возвратов за квартал)  
         19  item_count_sales7             197198 non-null  int64 (количество продаж за неделю)  
         20  item_count_sales30            197198 non-null  int64 (количество продаж за месяц)  
         21  item_count_sales90            197198 non-null  int64 (количество продаж за квартал)  
         22  item_count_returns7           197198 non-null  int64 (количество возвратов за неделю)  
         23  item_count_returns30          197198 non-null  int64 (количество возвратов за месяц)  
         24  item_count_returns90          197198 non-null  int64 (количество возвратов за квартал)  
         25  GmvTotal7                     187007 non-null  float64 --NaN (Gross Merchandise Value — валовой товарооборот. Это сумма всех продаж товаров за определённый период времени, без учёта возвратов, скидок, налогов и логистики.)
         26  GmvTotal30                    189268 non-null  float64 --NaN (Общая стоимость всех проданных товаров селлера за 30 дней)
         27  GmvTotal90                    189791 non-null  float64 --NaN (Общая стоимость всех проданных товаров селлера за 90 дней)
         28  ExemplarAcceptedCountTotal7   187007 non-null  float64 --NaN (Общее количество товаров в заказах у селлера за 7 дней)
         29  ExemplarAcceptedCountTotal30  189268 non-null  float64 --NaN (Общее количество товаров в заказах у селлера за 30 дней)
         30  ExemplarAcceptedCountTotal90  189791 non-null  float64 --NaN (Общее количество товаров в заказах у селлера за 90 дней) 
         31  OrderAcceptedCountTotal7      186797 non-null  float64 --NaN (Общее количество принятых заказов за 7 дней)
         32  OrderAcceptedCountTotal30     189038 non-null  float64 --NaN (Общее количество принятых заказов за 30 дней)
         33  OrderAcceptedCountTotal90     189681 non-null  float64 --NaN (Общее количество принятых заказов за 90 дней)
         34  ExemplarReturnedCountTotal7   187007 non-null  float64 --NaN (Общее количество возвращённых товаров селлера 7 дней)
         35  ExemplarReturnedCountTotal30  189268 non-null  float64 --NaN (Общее количество возвращённых товаров селлера 30 дней)
         36  ExemplarReturnedCountTotal90  189791 non-null  float64 --NaN (Общее количество возвращённых товаров селлера 90 дней)
         37  ExemplarReturnedValueTotal7   187007 non-null  float64 --NaN (Суммарная стоимость всех возвращённых товаров за последние 7 дней)
         38  ExemplarReturnedValueTotal30  189268 non-null  float64 --NaN (Суммарная стоимость всех возвращённых товаров за последние 30 дней)
         39  ExemplarReturnedValueTotal90  189791 non-null  float64 --NaN (Суммарная стоимость всех возвращённых товаров за последние 90 дней)
         40  ItemVarietyCount              196201 non-null  float64 --NaN (сколько SKU, моделей, цветов, размеров или других уникальных разновидностей одного товара представлено у продавца или на платформе)
         41  ItemAvailableCount            196201 non-null  float64 --NaN (Количество доступных (в наличии) экземпляров товара)
         42  seller_time_alive             197198 non-null  float64 (Сколько времени продавец активен на платформе)
         43  ItemID                        197198 non-null  int64  (ItemID → идентификатор самого товара, который может повторяться)
         44  SellerID                      197198 non-null  int64  
         dtypes: float64(27), int64(14), object(4)
         
            "Fake" (поддельные, фиктивные, подозрительные) возвраты — это возвраты, которые система определила как нетипичные или аномальные, например:        
            Возврат оформлен, но товар не был физически возвращён.        
            Возвраты с подозрительной периодичностью (например, мошенническая активность).        
            Возврат по схеме: купил — вернул — купил снова (например, ради бонусов или временного пользования).        
            Возврат товаров, не подлежащих возврату.
        """

        print(df_products_feedback_questions.shape)

        df_products_feedback_questions['description'].notna()

        print(df_products_feedback_questions[df_products_feedback_questions['description'].notna()]['description'].values)

        df_products_feedback_questions = df_products_feedback_questions.rename(columns={'CommercialTypeName4': 'CommercialCategory'})

        print(df_products_feedback_questions[df_products_feedback_questions['name_rus'].notna()]['name_rus'].values)

        print(df_products_feedback_questions)

        print(df_products_feedback_questions[df_products_feedback_questions['SellerID'].notna()]['SellerID'].values)

        print(df_products_feedback_questions[df_products_feedback_questions['brand_name'].notna()]['brand_name'].values)

        print(df_products_feedback_questions[df_products_feedback_questions['brand_name'].notna()])

        df_copy = df_products_feedback_questions.copy()

        print(df_copy)

        """##Проверка и исключение дубликатов"""

        df_copy.duplicated().sum()

        ##**Форматирование и приведение данных к правильному типу**

        df_copy.info()

        df_copy['brand_name'] = df_copy['brand_name'].astype('string')

        df_copy['name_rus'] = df_copy['name_rus'].astype('string')

        df_copy['description'] = df_copy['description'].astype('string')

        df_copy['CommercialCategory'] = df_copy['CommercialCategory'].astype('string')

        df_copy.info()

        """#Обработка пропущенных значений"""

        df_copy.isna().sum()

        """
        
        id	0
        resolution	0
        brand_name	80531
        description	26060
        name_rus	0
        CommercialCategory	0
        rating_1_count	150005
        rating_2_count	150005
        rating_3_count	150005
        rating_4_count	150005
        rating_5_count	150005
        comments_published_count	150005
        photos_published_count	150005
        videos_published_count	150005
        PriceDiscounted	0
        item_time_alive	0
        item_count_fake_returns7	0
        item_count_fake_returns30	0
        item_count_fake_returns90	0
        item_count_sales7	0
        item_count_sales30	0
        item_count_sales90	0
        item_count_returns7	0
        item_count_returns30	0
        item_count_returns90	0
        GmvTotal7	10191
        GmvTotal30	7930
        GmvTotal90	7407
        ExemplarAcceptedCountTotal7	10191
        ExemplarAcceptedCountTotal30	7930
        ExemplarAcceptedCountTotal90	7407
        OrderAcceptedCountTotal7	10401
        OrderAcceptedCountTotal30	8160
        OrderAcceptedCountTotal90	7517
        ExemplarReturnedCountTotal7	10191
        ExemplarReturnedCountTotal30	7930
        ExemplarReturnedCountTotal90	7407
        ExemplarReturnedValueTotal7	10191
        ExemplarReturnedValueTotal30	7930
        ExemplarReturnedValueTotal90	7407
        ItemVarietyCount	997
        ItemAvailableCount	997
        seller_time_alive	0
        ItemID	0
        SellerID	0
        
        """

        df_copy['brand_name'] = df_copy['brand_name'].fillna('')

        df_copy['description'] = df_copy['description'].fillna('')

        df_copy['rating_1_count'] = df_copy['rating_1_count'].fillna(0)

        df_copy['rating_2_count'] = df_copy['rating_2_count'].fillna(0)

        df_copy['rating_3_count'] = df_copy['rating_3_count'].fillna(0)

        df_copy['rating_4_count'] = df_copy['rating_4_count'].fillna(0)

        df_copy['rating_5_count'] = df_copy['rating_5_count'].fillna(0)

        df_copy['comments_published_count'] = df_copy['comments_published_count'].fillna(0)

        df_copy['photos_published_count'] = df_copy['photos_published_count'].fillna(0)

        df_copy['videos_published_count'] = df_copy['videos_published_count'].fillna(0)

        df_copy['GmvTotal7'] = df_copy['GmvTotal7'].fillna(0)

        df_copy['GmvTotal30'] = df_copy['GmvTotal30'].fillna(0)

        df_copy['GmvTotal90'] = df_copy['GmvTotal90'].fillna(0)

        df_copy['ExemplarAcceptedCountTotal7'] = df_copy['ExemplarAcceptedCountTotal7'].fillna(0)

        df_copy['ExemplarAcceptedCountTotal30'] = df_copy['ExemplarAcceptedCountTotal30'].fillna(0)

        df_copy['ExemplarAcceptedCountTotal90'] = df_copy['ExemplarAcceptedCountTotal90'].fillna(0)

        df_copy['OrderAcceptedCountTotal7'] = df_copy['OrderAcceptedCountTotal7'].fillna(0)

        df_copy['OrderAcceptedCountTotal30'] = df_copy['OrderAcceptedCountTotal30'].fillna(0)

        df_copy['OrderAcceptedCountTotal90'] = df_copy['OrderAcceptedCountTotal90'].fillna(0)

        df_copy['ExemplarReturnedCountTotal7'] = df_copy['ExemplarReturnedCountTotal7'].fillna(0)

        df_copy['ExemplarReturnedCountTotal30'] = df_copy['ExemplarReturnedCountTotal30'].fillna(0)

        df_copy['ExemplarReturnedCountTotal90'] = df_copy['ExemplarReturnedCountTotal90'].fillna(0)

        df_copy['ExemplarReturnedValueTotal7'] = df_copy['ExemplarReturnedValueTotal7'].fillna(0)

        df_copy['ExemplarReturnedValueTotal30'] = df_copy['ExemplarReturnedValueTotal30'].fillna(0)

        df_copy['ExemplarReturnedValueTotal90'] = df_copy['ExemplarReturnedValueTotal90'].fillna(0)

        df_copy['ItemVarietyCount'] = df_copy['ItemVarietyCount'].fillna(0)

        df_copy['ItemAvailableCount'] = df_copy['ItemAvailableCount'].fillna(0)

        df_copy.isna().sum()

        df_copy.head()

        df_copy.info()

        return df_copy


    """# Анализ данных
    
    **Этапы анализа:**
    1. Определить целевые и факторные переменные.
    2. Определить типы переменных.
    3. Проанализировать каждый признак отдельно.
    4. Проанализировать взаимосвязи признаков.
    5. Проинтерпретировать результаты.
    
    **Типы переменных:**
    1. Категориальные (например resolution, brand_name, SellerID, CommercialCategory).
    2. Количественные (rating_1_count, rating_2_count, rating_3_count, 
    rating_4_count, rating_5_count, comments_published_count, photos_published_count, 
    videos_published_count, PriceDiscounted, item_count_fake_returns7, item_count_fake_returns30, 
    item_count_fake_returns90, item_count_sales7, item_count_sales30, item_count_sales90,
    item_count_returns7, item_count_returns30, item_count_returns90, GmvTotal7, GmvTotal30,
    GmvTotal90, ExemplarAcceptedCountTotal7, ExemplarAcceptedCountTotal30, ExemplarAcceptedCountTotal90,
    OrderAcceptedCountTotal7, OrderAcceptedCountTotal30, OrderAcceptedCountTotal90,
    ExemplarReturnedCountTotal7, ExemplarReturnedCountTotal30, ExemplarReturnedCountTotal90,
    ExemplarReturnedValueTotal7, ExemplarReturnedValueTotal30, ExemplarReturnedValueTotal90,
    ItemVarietyCount, ItemAvailableCount, item_time_alive, seller_time_alive)
    3. Текстовые (например, name_rus, description).
    
    
    ## Обзор и описание данных
    
    ###Количественные призаки (rating_1_count, rating_2_count, rating_3_count, 
    rating_4_count, rating_5_count, comments_published_count, photos_published_count, 
    videos_published_count, PriceDiscounted, item_count_fake_returns7, item_count_fake_returns30, 
    item_count_fake_returns90, item_count_sales7, item_count_sales30, item_count_sales90,
    item_count_returns7, item_count_returns30, item_count_returns90, GmvTotal7, GmvTotal30,
    GmvTotal90, ExemplarAcceptedCountTotal7, ExemplarAcceptedCountTotal30, ExemplarAcceptedCountTotal90,
    OrderAcceptedCountTotal7, OrderAcceptedCountTotal30, OrderAcceptedCountTotal90,
    ExemplarReturnedCountTotal7, ExemplarReturnedCountTotal30, ExemplarReturnedCountTotal90,
    ExemplarReturnedValueTotal7, ExemplarReturnedValueTotal30, ExemplarReturnedValueTotal90,
    ItemVarietyCount, ItemAvailableCount, item_time_alive, seller_time_alive)
    """
    def rating_1_count_analitics(self, df_copy):
        rating_1_count = df_copy.rating_1_count

        max_value = rating_1_count.max()
        min_value = rating_1_count.min()
        mean_value = rating_1_count.mean()
        median_value = rating_1_count.median()
        print(f'Наибольшее количество рейтинга 1: {max_value}', f'Наименьшее количество рейтинга 1: {min_value}',
              f'Среднее количество рейтинга 1: {mean_value}', f'Медианное количество рейтинга 1: {median_value}', sep='\n')

        percentile_10_value = rating_1_count.quantile(0.10)
        percentile_25_value = rating_1_count.quantile(0.25)
        percentile_50_value = rating_1_count.quantile(0.50)
        percentile_75_value = rating_1_count.quantile(0.75)
        percentile_90_value = rating_1_count.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        rating_1_count.describe()

        sns.histplot(rating_1_count, bins=20, color='blue')
        plt.title('Гистограмма распределения рейтинга 1 товаров')
        plt.xlabel('Рейтинг 1')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        len(df_copy[df_copy.rating_1_count > percentile_90_value])

        sns.histplot(rating_1_count[rating_1_count > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения рейтинга 1 товаров')
        plt.xlabel('Рейтинг 1')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(rating_1_count[rating_1_count > percentile_90_value], vert=False)
        plt.title('Боксплот рейтинга 1 товаров')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = rating_1_count.skew()
        kurtosis = rating_1_count.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.rating_1_count)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение рейтинга 1 по исходной выборке - 0, максимальное - 1007. Размах значений составил 1007. Рекомендуется дополнительно проанализировать товары с самым низким рейтингом 1.
        2. Cреднее значение рейтинга составляет примерно 0.4902, а медианное - 0.0. Сдвиг незначительный, но свидетельствует о скошенности распределения вправо, рекомендуется проанализировать дополнительно.
        3. Рейтинг товаров до 90% не превышает 0, что свидетельствует о том, что у большинства товаров нет рейтинга 1.
        4. На гистограмме видно, что распределение признака отлично от нормального.Боксплот не показывает наличие оставшихся выбросов, и он изначально строится после 90 процентиля, если построить боксплот по всем данным, то можно заметить нулевой боксплот, что также свидетельствует об отличии распределения признака от нормального.
        5. Коэффициент ассиметрии больше 1, что так же свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 67296.0537 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def rating_2_count_analitics(self, df_copy):
        rating_2_count = df_copy.rating_2_count

        max_value = rating_2_count.max()
        min_value = rating_2_count.min()
        mean_value = rating_2_count.mean()
        median_value = rating_2_count.median()
        print(f'Наибольшее количество рейтинга 2: {max_value}', f'Наименьшее количество рейтинга 2: {min_value}',
              f'Среднее количество рейтинга 2: {mean_value}', f'Медианное количество рейтинга 2: {median_value}', sep='\n')

        percentile_10_value = rating_2_count.quantile(0.10)
        percentile_25_value = rating_2_count.quantile(0.25)
        percentile_50_value = rating_2_count.quantile(0.50)
        percentile_75_value = rating_2_count.quantile(0.75)
        percentile_90_value = rating_2_count.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        rating_2_count.describe()

        sns.histplot(rating_2_count, bins=20, color='blue')
        plt.title('Гистограмма распределения рейтинга 2 товаров')
        plt.xlabel('Рейтинг 2')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.rating_2_count > percentile_90_value]))

        sns.histplot(rating_2_count[rating_2_count > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения рейтинга 2 товаров')
        plt.xlabel('Рейтинг 2')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(rating_2_count[rating_2_count > percentile_90_value], vert=False)
        plt.title('Боксплот рейтинга 2 товаров')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = rating_2_count.skew()
        kurtosis = rating_2_count.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.rating_2_count)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение рейтинга 2 по исходной выборке - 0, максимальное - 199. Размах значений составил 199. Рекомендуется дополнительно проанализировать товары с низким рейтингом 2.
        2. Cреднее значение рейтинга составляет примерно 0.1423, а медианное - 0.0. Сдвиг незначительный, но свидетельствует о скошенности распределения вправо, рекомендуется проанализировать дополнительно.
        3. Рейтинг товаров до 90% не превышает 0, что свидетельствует о том, что у большинства товаров нет рейтинга 2.
        4. На гистограмме видно, что распределение признака отлично от нормального.Боксплот не показывает наличие оставшихся выбросов, и он изначально строится после 90 процентиля, если построить боксплот по всем данным, то можно заметить нулевой боксплот, что также свидетельствует об отличии распределения признака от нормального.
        5. Коэффициент ассиметрии больше 1, что так же свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 67461.75 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def rating_3_count_analitics(self, df_copy):
        rating_3_count = df_copy.rating_3_count

        max_value = rating_3_count.max()
        min_value = rating_3_count.min()
        mean_value = rating_3_count.mean()
        median_value = rating_3_count.median()
        print(f'Наибольшее количество рейтинга 3: {max_value}', f'Наименьшее количество рейтинга 3: {min_value}',
              f'Среднее количество рейтинга 3: {mean_value}', f'Медианное количество рейтинга 3: {median_value}', sep='\n')

        percentile_10_value = rating_3_count.quantile(0.10)
        percentile_25_value = rating_3_count.quantile(0.25)
        percentile_50_value = rating_3_count.quantile(0.50)
        percentile_75_value = rating_3_count.quantile(0.75)
        percentile_90_value = rating_3_count.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        rating_3_count.describe()

        sns.histplot(rating_3_count, bins=20, color='blue')
        plt.title('Гистограмма распределения рейтинга 3 товаров')
        plt.xlabel('Рейтинг 3')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.rating_3_count > percentile_90_value]))

        sns.histplot(rating_3_count[rating_3_count > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения рейтинга 3 товаров')
        plt.xlabel('Рейтинг 3')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(rating_3_count[rating_3_count > percentile_90_value], vert=False)
        plt.title('Боксплот рейтинга 3 товаров')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = rating_3_count.skew()
        kurtosis = rating_3_count.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.rating_3_count)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение рейтинга 3 по исходной выборке - 0, максимальное - 329. Размах значений составил 329. Рекомендуется дополнительно проанализировать товары с рейтингом 3.
        2. Cреднее значение рейтинга составляет примерно 0.2656, а медианное - 0.0. Сдвиг незначительный, но свидетельствует о скошенности распределения вправо, рекомендуется проанализировать дополнительно.
        3. Рейтинг товаров до 90% не превышает 0, что свидетельствует о том, что у большинства товаров нет рейтинга 3.
        4. На гистограмме видно, что распределение признака отлично от нормального.Боксплот не показывает наличие оставшихся выбросов, и он изначально строится после 90 процентиля, если построить боксплот по всем данным, то можно заметить нулевой боксплот, что также свидетельствует об отличии распределения признака от нормального.
        5. Коэффициент ассиметрии больше 1, что так же свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 66522.976 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """
    def rating_4_count_analitics(self, df_copy):
        rating_4_count = df_copy.rating_4_count

        max_value = rating_4_count.max()
        min_value = rating_4_count.min()
        mean_value = rating_4_count.mean()
        median_value = rating_4_count.median()
        print(f'Наибольшее количество рейтинга 4: {max_value}', f'Наименьшее количество рейтинга 4: {min_value}',
              f'Среднее количество рейтинга 4: {mean_value}', f'Медианное количество рейтинга 4: {median_value}', sep='\n')

        percentile_10_value = rating_4_count.quantile(0.10)
        percentile_25_value = rating_4_count.quantile(0.25)
        percentile_50_value = rating_4_count.quantile(0.50)
        percentile_75_value = rating_4_count.quantile(0.75)
        percentile_90_value = rating_4_count.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        rating_4_count.describe()

        sns.histplot(rating_4_count, bins=20, color='blue')
        plt.title('Гистограмма распределения рейтинга 4 товаров')
        plt.xlabel('Рейтинг 4')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.rating_4_count > percentile_90_value]))

        sns.histplot(rating_4_count[rating_4_count > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения рейтинга 4 товаров')
        plt.xlabel('Рейтинг 4')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(rating_4_count[rating_4_count > percentile_90_value], vert=False)
        plt.title('Боксплот рейтинга 4 товаров')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = rating_4_count.skew()
        kurtosis = rating_4_count.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.rating_4_count)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение рейтинга 4 по исходной выборке - 0, максимальное - 518. Размах значений составил 518. Рекомендуется дополнительно проанализировать товары с рейтингом 4.
        2. Cреднее значение рейтинга составляет примерно 0.2998, а медианное - 0.0. Сдвиг незначительный, но свидетельствует о скошенности распределения вправо, рекомендуется проанализировать дополнительно.
        3. Рейтинг товаров до 90% не превышает 0, что свидетельствует о том, что у большинства товаров нет рейтинга 4.
        4. На гистограмме видно, что распределение признака отлично от нормального.Боксплот не показывает наличие оставшихся выбросов, и он изначально строится после 90 процентиля, если построить боксплот по всем данным, то можно заметить нулевой боксплот, что также свидетельствует об отличии распределения признака от нормального.
        5. Коэффициент ассиметрии больше 1, что так же свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 65792.13 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def rating_5_count_analitics(self, df_copy):
        rating_5_count = df_copy.rating_5_count

        max_value = rating_5_count.max()
        min_value = rating_5_count.min()
        mean_value = rating_5_count.mean()
        median_value = rating_5_count.median()
        print(f'Наибольшее количество рейтинга 5: {max_value}', f'Наименьшее количество рейтинга 5: {min_value}',
              f'Среднее количество рейтинга 5: {mean_value}', f'Медианное количество рейтинга 5: {median_value}', sep='\n')

        percentile_10_value = rating_5_count.quantile(0.10)
        percentile_25_value = rating_5_count.quantile(0.25)
        percentile_50_value = rating_5_count.quantile(0.50)
        percentile_75_value = rating_5_count.quantile(0.75)
        percentile_90_value = rating_5_count.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        rating_5_count.describe()

        sns.histplot(rating_5_count, bins=20, color='blue')
        plt.title('Гистограмма распределения рейтинга 5 товаров')
        plt.xlabel('Рейтинг 5')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.rating_5_count > percentile_90_value]))

        sns.histplot(rating_5_count[rating_5_count > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения рейтинга 5 товаров')
        plt.xlabel('Рейтинг 5')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(rating_5_count[rating_5_count > percentile_90_value], vert=False)
        plt.title('Боксплот рейтинга 5 товаров')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = rating_5_count.skew()
        kurtosis = rating_5_count.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.rating_5_count)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение рейтинга 5 по исходной выборке - 0, максимальное - 4465. Размах значений составил 4465. Рекомендуется дополнительно проанализировать товары с рейтингом 5.
        2. Cреднее значение рейтинга составляет примерно 3.2179, а медианное - 0.0. Сдвиг значительный.
        3. Рейтинг товаров до 75% не превышает 0, что свидетельствует о том, что у большинства товаров нет рейтинга 5.
        4. На гистограмме видно, что распределение признака отлично от нормального.Боксплот не показывает наличие оставшихся выбросов, и он изначально строится после 90 процентиля, если построить боксплот по всем данным, то можно заметить нулевой боксплот, что также свидетельствует об отличии распределения признака от нормального.
        5. Коэффициент ассиметрии больше 1, что так же свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 64649.205 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def comments_published_count_analitics(self, df_copy):
        comments_published_count = df_copy.comments_published_count

        max_value = comments_published_count.max()
        min_value = comments_published_count.min()
        mean_value = comments_published_count.mean()
        median_value = comments_published_count.median()
        print(f'Наибольшее количество комментариев: {max_value}', f'Наименьшее количество комментариев: {min_value}',
              f'Среднее количество комментариев: {mean_value}', f'Медианное количество комментариев: {median_value}', sep='\n')

        percentile_10_value = comments_published_count.quantile(0.10)
        percentile_25_value = comments_published_count.quantile(0.25)
        percentile_50_value = comments_published_count.quantile(0.50)
        percentile_75_value = comments_published_count.quantile(0.75)
        percentile_90_value = comments_published_count.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        comments_published_count.describe()

        sns.histplot(comments_published_count, bins=20, color='blue')
        plt.title('Гистограмма распределения количества комментариев товаров')
        plt.xlabel('Количество комментариев')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.comments_published_count > percentile_90_value]))

        sns.histplot(comments_published_count[comments_published_count > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения количества комментариев товаров')
        plt.xlabel('Количество комментариев')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(comments_published_count[comments_published_count > percentile_90_value], vert=False)
        plt.title('Боксплот количества комментариев товаров')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = comments_published_count.skew()
        kurtosis = comments_published_count.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.comments_published_count)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение количества комментариев по исходной выборке - 0, максимальное - 1753. Размах значений составил 1753. Рекомендуется дополнительно проанализировать товары с большим количеством комментариев.
        2. Cреднее значение рейтинга составляет примерно 2.2781, а медианное - 0.0. Сдвиг значительный.
        3. Рейтинг товаров до 75% не превышает 0, что свидетельствует о том, что у большинства товаров нет комментариев.
        4. На гистограмме видно, что распределение признака отлично от нормального.Боксплот не показывает наличие оставшихся выбросов, и он изначально строится после 90 процентиля, если построить боксплот по всем данным, то можно заметить нулевой боксплот, что также свидетельствует об отличии распределения признака от нормального.
        5. Коэффициент ассиметрии больше 1, что так же свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 66176.68 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def photos_published_count_analitics(self, df_copy):
        photos_published_count = df_copy.photos_published_count

        max_value = photos_published_count.max()
        min_value = photos_published_count.min()
        mean_value = photos_published_count.mean()
        median_value = photos_published_count.median()
        print(f'Наибольшее количество фото: {max_value}', f'Наименьшее количество фото: {min_value}',
              f'Среднее количество фото: {mean_value}', f'Медианное количество фото: {median_value}', sep='\n')

        percentile_10_value = photos_published_count.quantile(0.10)
        percentile_25_value = photos_published_count.quantile(0.25)
        percentile_50_value = photos_published_count.quantile(0.50)
        percentile_75_value = photos_published_count.quantile(0.75)
        percentile_90_value = photos_published_count.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        photos_published_count.describe()

        sns.histplot(photos_published_count, bins=20, color='blue')
        plt.title('Гистограмма распределения количества фото товаров')
        plt.xlabel('Количество фото')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.photos_published_count > percentile_90_value]))

        sns.histplot(photos_published_count[photos_published_count > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения количества фото товаров')
        plt.xlabel('Количество фото')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(photos_published_count[photos_published_count > percentile_90_value], vert=False)
        plt.title('Боксплот количества фото товаров')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = photos_published_count.skew()
        kurtosis = photos_published_count.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.photos_published_count)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение количества фото в комментариеях по исходной выборке - 0, максимальное - 1035. Размах значений составил 1035. Рекомендуется дополнительно проанализировать товары с большим количеством фото в комментариях.
        2. Cреднее значение рейтинга составляет примерно 1.05592, а медианное - 0.0. Сдвиг значительный.
        3. Рейтинг товаров до 75% не превышает 0, что свидетельствует о том, что у большинства товаров нет фото в комментариях.
        4. На гистограмме видно, что распределение признака отлично от нормального.Боксплот не показывает наличие оставшихся выбросов, и он изначально строится после 90 процентиля, если построить боксплот по всем данным, то можно заметить нулевой боксплот, что также свидетельствует об отличии распределения признака от нормального.
        5. Коэффициент ассиметрии больше 1, что так же свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 64260.79 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def videos_published_count_analitics(self, df_copy):
        videos_published_count = df_copy.videos_published_count

        max_value = videos_published_count.max()
        min_value = videos_published_count.min()
        mean_value = videos_published_count.mean()
        median_value = videos_published_count.median()
        print(f'Наибольшее количество видео: {max_value}', f'Наименьшее количество видео: {min_value}',
              f'Среднее количество видео: {mean_value}', f'Медианное количество видео: {median_value}', sep='\n')

        percentile_10_value = videos_published_count.quantile(0.10)
        percentile_25_value = videos_published_count.quantile(0.25)
        percentile_50_value = videos_published_count.quantile(0.50)
        percentile_75_value = videos_published_count.quantile(0.75)
        percentile_90_value = videos_published_count.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        videos_published_count.describe()

        sns.histplot(videos_published_count, bins=20, color='blue')
        plt.title('Гистограмма распределения количества видео в комментариях товаров')
        plt.xlabel('Количество видео в комментариях')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.videos_published_count > percentile_90_value]))

        sns.histplot(videos_published_count[videos_published_count > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения количества видео в комментариях товаров')
        plt.xlabel('Количество видео в комментариях')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(videos_published_count[videos_published_count > percentile_90_value], vert=False)
        plt.title('Боксплот количества видео в комментария товаров')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = videos_published_count.skew()
        kurtosis = videos_published_count.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.videos_published_count)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение количества фото в комментариеях по исходной выборке - 0, максимальное - 241. Размах значений составил 241. Рекомендуется дополнительно проанализировать товары с большим количеством видео в комментариях.
        2. Cреднее значение рейтинга составляет примерно 0.1155, а медианное - 0.0. Сдвиг значительный.
        3. Рейтинг товаров до 90% не превышает 0, что свидетельствует о том, что у большинства товаров нет видео в комментариях.
        4. На гистограмме видно, что распределение признака отлично от нормального.Боксплот не показывает наличие оставшихся выбросов, и он изначально строится после 90 процентиля, если построить боксплот по всем данным, то можно заметить нулевой боксплот, что также свидетельствует об отличии распределения признака от нормального.
        5. Коэффициент ассиметрии больше 1, что так же свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 70191.19 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def PriceDiscounted_analitics(self, df_copy):
        PriceDiscounted = df_copy.PriceDiscounted

        max_value = PriceDiscounted.max()
        min_value = PriceDiscounted.min()
        mean_value = PriceDiscounted.mean()
        median_value = PriceDiscounted.median()
        print(f'Наибольшая цена: {max_value}', f'Наименьшая цена: {min_value}',
              f'Средняя цена: {mean_value}', f'Медианная цена: {median_value}', sep='\n')

        percentile_10_value = PriceDiscounted.quantile(0.10)
        percentile_25_value = PriceDiscounted.quantile(0.25)
        percentile_50_value = PriceDiscounted.quantile(0.50)
        percentile_75_value = PriceDiscounted.quantile(0.75)
        percentile_90_value = PriceDiscounted.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        PriceDiscounted.describe()

        sns.histplot(PriceDiscounted, bins=20, color='blue')
        plt.title('Гистограмма распределения кцены товаров')
        plt.xlabel('Цена товара')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(PriceDiscounted, vert=False)
        plt.title('Боксплот цены товаров')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = PriceDiscounted.skew()
        kurtosis = PriceDiscounted.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.PriceDiscounted)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение количества фото в комментариеях по исходной выборке - 0, максимальное - 1816.56. Размах значений составил 1816.56. Рекомендуется дополнительно проанализировать товары с большой ценой.
        2. Cреднее значение цены составляет примерно 758.97, а медианное - 736.695. Сдвиг незначительный.
        3. 10-й процентиль: 593.7581624476625
        25-й процентиль: 651.0815159196782
        50-й процентиль: 736.6954210210963
        75-й процентиль: 827.9551699189296
        90-й процентиль: 962.3647125978565
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии меньше 1, что не свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 3590.3 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def item_count_fake_returns7_analitics(self, df_copy):
        item_count_fake_returns7 = df_copy.item_count_fake_returns7

        max_value = item_count_fake_returns7.max()
        min_value = item_count_fake_returns7.min()
        mean_value = item_count_fake_returns7.mean()
        median_value = item_count_fake_returns7.median()
        print(f'Наибольшее количество фиктивных возвратов за неделю: {max_value}', f'Наименьшее количество фиктивных возвратов за неделю: {min_value}',
              f'Среднее количество фиктивных возвратов за неделю: {mean_value}', f'Медианное количество фиктивных возвратов за неделю: {median_value}', sep='\n')

        percentile_10_value = item_count_fake_returns7.quantile(0.10)
        percentile_25_value = item_count_fake_returns7.quantile(0.25)
        percentile_50_value = item_count_fake_returns7.quantile(0.50)
        percentile_75_value = item_count_fake_returns7.quantile(0.75)
        percentile_90_value = item_count_fake_returns7.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        item_count_fake_returns7.describe()

        sns.histplot(item_count_fake_returns7, bins=20, color='blue')
        plt.title('Гистограмма распределения количества фиктивных возвратов за неделю')
        plt.xlabel('Количество фиктивных возвратов за неделю')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.item_count_fake_returns7 > percentile_90_value]))

        sns.histplot(item_count_fake_returns7[item_count_fake_returns7 > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения количества фиктивных возвратов за неделюв')
        plt.xlabel('Количество фиктивных возвратов за неделю')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(item_count_fake_returns7[item_count_fake_returns7 > percentile_90_value], vert=False)
        plt.title('Боксплот количества фиктивных возвратов за неделю')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = item_count_fake_returns7.skew()
        kurtosis = item_count_fake_returns7.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.item_count_fake_returns7)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение количества фото в комментариеях по исходной выборке - 0, максимальное - 30. Размах значений составил 30. Рекомендуется дополнительно проанализировать товары с наибольшим количеством фиктивных возвратов за неделю.
        2. Cреднее значение цены составляет примерно 0.00752, а медианное - 0.0. Сдвиг незначительный.
        3. Рейтинг товаров до 90% не превышает 0, что свидетельствует о том, что у большинства товаров нет фиктивных возвратов за неделю.
        4. На гистограмме видно, что распределение признака не похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии меньше 1, что не свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 75252.25 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def item_count_fake_returns30_analitics(self, df_copy):
        item_count_fake_returns30 = df_copy.item_count_fake_returns30

        max_value = item_count_fake_returns30.max()
        min_value = item_count_fake_returns30.min()
        mean_value = item_count_fake_returns30.mean()
        median_value = item_count_fake_returns30.median()
        print(f'Наибольшее количество фиктивных возвратов за месяц: {max_value}', f'Наименьшее количество фиктивных возвратов за месяц: {min_value}',
              f'Среднее количество фиктивных возвратов за месяц: {mean_value}', f'Медианное количество фиктивных возвратов за месяц: {median_value}', sep='\n')

        percentile_10_value = item_count_fake_returns30.quantile(0.10)
        percentile_25_value = item_count_fake_returns30.quantile(0.25)
        percentile_50_value = item_count_fake_returns30.quantile(0.50)
        percentile_75_value = item_count_fake_returns30.quantile(0.75)
        percentile_90_value = item_count_fake_returns30.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        item_count_fake_returns30.describe()

        sns.histplot(item_count_fake_returns30, bins=20, color='blue')
        plt.title('Гистограмма распределения количества фиктивных возвратов за месяц')
        plt.xlabel('Количество фиктивных возвратов за месяц')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.item_count_fake_returns30 > percentile_90_value]))

        sns.histplot(item_count_fake_returns30[item_count_fake_returns30 > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения количества фиктивных возвратов за месяц')
        plt.xlabel('Количество фиктивных возвратов за месяц')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(item_count_fake_returns30[item_count_fake_returns30 > percentile_90_value], vert=False)
        plt.title('Боксплот количества фиктивных возвратов за месяц')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = item_count_fake_returns30.skew()
        kurtosis = item_count_fake_returns30.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.item_count_fake_returns30)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение количества фото в комментариеях по исходной выборке - 0, максимальное - 103. Размах значений составил 103. Рекомендуется дополнительно проанализировать товары с наибольшим количеством фиктивных возвратов за месяц.
        2. Cреднее значение цены составляет примерно 0.02585, а медианное - 0.0. Сдвиг незначительный.
        3. Рейтинг товаров до 90% не превышает 0, что свидетельствует о том, что у большинства товаров нет фиктивных возвратов за месяц.
        4. На гистограмме видно, что распределение признака не похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии меньше 1, что не свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 73619.606 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def item_count_fake_returns90_analitics(self, df_copy):
        item_count_fake_returns90 = df_copy.item_count_fake_returns90

        max_value = item_count_fake_returns90.max()
        min_value = item_count_fake_returns90.min()
        mean_value = item_count_fake_returns90.mean()
        median_value = item_count_fake_returns90.median()
        print(f'Наибольшее количество фиктивных возвратов за квартал: {max_value}', f'Наименьшее количество фиктивных возвратов за квартал: {min_value}',
              f'Среднее количество фиктивных возвратов за квартал: {mean_value}', f'Медианное количество фиктивных возвратов за квартал: {median_value}', sep='\n')

        percentile_10_value = item_count_fake_returns90.quantile(0.10)
        percentile_25_value = item_count_fake_returns90.quantile(0.25)
        percentile_50_value = item_count_fake_returns90.quantile(0.50)
        percentile_75_value = item_count_fake_returns90.quantile(0.75)
        percentile_90_value = item_count_fake_returns90.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        item_count_fake_returns90.describe()

        sns.histplot(item_count_fake_returns90, bins=20, color='blue')
        plt.title('Гистограмма распределения количества фиктивных возвратов за квартал')
        plt.xlabel('Количество фиктивных возвратов за квартал')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.item_count_fake_returns90 > percentile_90_value]))

        sns.histplot(item_count_fake_returns90[item_count_fake_returns90 > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения количества фиктивных возвратов за квартал')
        plt.xlabel('Количество фиктивных возвратов за квартал')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(item_count_fake_returns90[item_count_fake_returns90 > percentile_90_value], vert=False)
        plt.title('Боксплот количества фиктивных возвратов за квартал')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = item_count_fake_returns90.skew()
        kurtosis = item_count_fake_returns90.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.item_count_fake_returns90)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение количества фото в комментариеях по исходной выборке - 0, максимальное - 148. Размах значений составил 148. Рекомендуется дополнительно проанализировать товары с наибольшим количеством фиктивных возвратов за квартал.
        2. Cреднее значение цены составляет примерно 0.062607, а медианное - 0.0. Сдвиг незначительный.
        3. Рейтинг товаров до 90% не превышает 0, что свидетельствует о том, что у большинства товаров нет фиктивных возвратов за квартал.
        4. На гистограмме видно, что распределение признака не похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии меньше 1, что не свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 72314.54 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def item_count_sales7_analitics(self, df_copy):
        item_count_sales7 = df_copy.item_count_sales7

        max_value = item_count_sales7.max()
        min_value = item_count_sales7.min()
        mean_value = item_count_sales7.mean()
        median_value = item_count_sales7.median()
        print(f'Наибольшее количество продаж за неделю: {max_value}', f'Наименьшее количество продаж за неделю: {min_value}',
              f'Среднее количество продаж за неделю: {mean_value}', f'Медианное количество продаж за неделю: {median_value}', sep='\n')

        percentile_10_value = item_count_sales7.quantile(0.10)
        percentile_25_value = item_count_sales7.quantile(0.25)
        percentile_50_value = item_count_sales7.quantile(0.50)
        percentile_75_value = item_count_sales7.quantile(0.75)
        percentile_90_value = item_count_sales7.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        item_count_sales7.describe()

        sns.histplot(item_count_sales7, bins=20, color='blue')
        plt.title('Гистограмма распределения количества продаж за неделю')
        plt.xlabel('Количество продаж за неделю')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.item_count_sales7 > percentile_90_value]))

        sns.histplot(item_count_sales7[item_count_sales7 > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения количества продаж за неделю')
        plt.xlabel('Количество продаж за неделю')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(item_count_sales7[item_count_sales7 > percentile_90_value], vert=False)
        plt.title('Боксплот количества продаж за неделю')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = item_count_sales7.skew()
        kurtosis = item_count_sales7.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.item_count_sales7)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение количества фото в комментариеях по исходной выборке - 0, максимальное - 4257. Размах значений составил 4257. Рекомендуется дополнительно проанализировать товары с наибольшим количеством продаж за неделю.
        2. Cреднее значение цены составляет примерно 0.68176, а медианное - 0.0. Сдвиг незначительный.
        3. Рейтинг товаров до 90% не превышает 0, что свидетельствует о том, что у большинства товаров нет продаж за неделю.
        4. На гистограмме видно, что распределение признака не похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии меньше 1, что не свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 70088.579 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def item_count_sales30_analitics(self, df_copy):
        item_count_sales30 = df_copy.item_count_sales30

        max_value = item_count_sales30.max()
        min_value = item_count_sales30.min()
        mean_value = item_count_sales30.mean()
        median_value = item_count_sales30.median()
        print(f'Наибольшее количество продаж за месяц: {max_value}', f'Наименьшее количество продаж за месяц: {min_value}',
              f'Среднее количество продаж за месяц: {mean_value}', f'Медианное количество продаж за месяц: {median_value}', sep='\n')

        percentile_10_value = item_count_sales30.quantile(0.10)
        percentile_25_value = item_count_sales30.quantile(0.25)
        percentile_50_value = item_count_sales30.quantile(0.50)
        percentile_75_value = item_count_sales30.quantile(0.75)
        percentile_90_value = item_count_sales30.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        item_count_sales30.describe()

        sns.histplot(item_count_sales30, bins=20, color='blue')
        plt.title('Гистограмма распределения количества продаж за месяц')
        plt.xlabel('Количество продаж за месяц')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.item_count_sales30 > percentile_90_value]))

        sns.histplot(item_count_sales30[item_count_sales30 > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения количества продаж за месяц')
        plt.xlabel('Количество продаж за месяц')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(item_count_sales30[item_count_sales30 > percentile_90_value], vert=False)
        plt.title('Боксплот количества продаж за месяц')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = item_count_sales30.skew()
        kurtosis = item_count_sales30.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.item_count_sales30)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение количества фото в комментариеях по исходной выборке - 0, максимальное - 14558. Размах значений составил 14558. Рекомендуется дополнительно проанализировать товары с наибольшим количеством продаж за месяц.
        2. Cреднее значение цены составляет примерно 2.373, а медианное - 0.0. Сдвиг незначительный.
        3. Рейтинг товаров до 90% не превышает 0, что свидетельствует о том, что у большинства товаров нет продаж за месяц.
        4. На гистограмме видно, что распределение признака не похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии меньше 1, что не свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 70362.249 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def item_count_sales90_analitics(self, df_copy):
        item_count_sales90 = df_copy.item_count_sales90

        max_value = item_count_sales90.max()
        min_value = item_count_sales90.min()
        mean_value = item_count_sales90.mean()
        median_value = item_count_sales90.median()
        print(f'Наибольшее количество продаж за квартал: {max_value}', f'Наименьшее количество продаж за квартал: {min_value}',
              f'Среднее количество продаж за квартал: {mean_value}', f'Медианное количество продаж за квартал: {median_value}', sep='\n')

        percentile_10_value = item_count_sales90.quantile(0.10)
        percentile_25_value = item_count_sales90.quantile(0.25)
        percentile_50_value = item_count_sales90.quantile(0.50)
        percentile_75_value = item_count_sales90.quantile(0.75)
        percentile_90_value = item_count_sales90.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        item_count_sales90.describe()

        sns.histplot(item_count_sales90, bins=20, color='blue')
        plt.title('Гистограмма распределения количества продаж за квартал')
        plt.xlabel('Количество продаж за квартал')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.item_count_sales90 > percentile_90_value]))

        sns.histplot(item_count_sales90[item_count_sales90 > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения количества продаж за квартал')
        plt.xlabel('Количество продаж за квартал')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(item_count_sales90[item_count_sales90 > percentile_90_value], vert=False)
        plt.title('Боксплот количества продаж за квартал')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = item_count_sales90.skew()
        kurtosis = item_count_sales90.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.item_count_sales90)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение количества фото в комментариеях по исходной выборке - 0, максимальное - 19877. Размах значений составил 19877. Рекомендуется дополнительно проанализировать товары с наибольшим количеством продаж за квартал.
        2. Cреднее значение цены составляет примерно 5.8099, а медианное - 0.0. Сдвиг значительный.
        3. Рейтинг товаров до 75% не превышает 0, что свидетельствует о том, что у большинства товаров нет продаж за квартал.
        4. На гистограмме видно, что распределение признака не похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии меньше 1, что не свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 769178.46 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def item_count_returns7_analitics(self, df_copy):
        item_count_returns7 = df_copy.item_count_returns7

        max_value = item_count_returns7.max()
        min_value = item_count_returns7.min()
        mean_value = item_count_returns7.mean()
        median_value = item_count_returns7.median()
        print(f'Наибольшее количество возвратов за неделю: {max_value}', f'Наименьшее количество возвратов за неделю: {min_value}',
              f'Среднее количество возвратов за неделю: {mean_value}', f'Медианное количество возвратов за неделю: {median_value}', sep='\n')

        percentile_10_value = item_count_returns7.quantile(0.10)
        percentile_25_value = item_count_returns7.quantile(0.25)
        percentile_50_value = item_count_returns7.quantile(0.50)
        percentile_75_value = item_count_returns7.quantile(0.75)
        percentile_90_value = item_count_returns7.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        item_count_returns7.describe()

        sns.histplot(item_count_returns7, bins=20, color='blue')
        plt.title('Гистограмма распределения количества возвратов за неделю')
        plt.xlabel('Количество возвратов за неделю')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.item_count_returns7 > percentile_90_value]))

        sns.histplot(item_count_returns7[item_count_returns7 > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения количества возвратов за неделю')
        plt.xlabel('Количество возвратов за неделю')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(item_count_returns7[item_count_returns7 > percentile_90_value], vert=False)
        plt.title('Боксплот количества возвратов за неделю')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = item_count_returns7.skew()
        kurtosis = item_count_returns7.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.item_count_returns7)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение количества фото в комментариеях по исходной выборке - 0, максимальное - 43. Размах значений составил 43. Рекомендуется дополнительно проанализировать товары с наибольшим количеством возвратов за неделю.
        2. Cреднее значение цены составляет примерно 0.021, а медианное - 0.0. Сдвиг незначительный.
        3. Рейтинг товаров до 90% не превышает 0, что свидетельствует о том, что у большинства товаров нет возвратов за неделю.
        4. На гистограмме видно, что распределение признака не похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии меньше 1, что не свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 73501.14 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def item_count_returns30_analitics(self, df_copy):
        item_count_returns30 = df_copy.item_count_returns30

        max_value = item_count_returns30.max()
        min_value = item_count_returns30.min()
        mean_value = item_count_returns30.mean()
        median_value = item_count_returns30.median()
        print(f'Наибольшее количество возвратов за месяц: {max_value}', f'Наименьшее количество возвратов за месяц: {min_value}',
              f'Среднее количество возвратов за месяц: {mean_value}', f'Медианное количество возвратов за месяц: {median_value}', sep='\n')

        percentile_10_value = item_count_returns30.quantile(0.10)
        percentile_25_value = item_count_returns30.quantile(0.25)
        percentile_50_value = item_count_returns30.quantile(0.50)
        percentile_75_value = item_count_returns30.quantile(0.75)
        percentile_90_value = item_count_returns30.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        item_count_returns30.describe()

        sns.histplot(item_count_returns30, bins=20, color='blue')
        plt.title('Гистограмма распределения количества возвратов за месяц')
        plt.xlabel('Количество возвратов за месяц')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.item_count_returns30 > percentile_90_value]))

        sns.histplot(item_count_returns30[item_count_returns30 > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения количества возвратов за месяц')
        plt.xlabel('Количество возвратов за месяц')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(item_count_returns30[item_count_returns30 > percentile_90_value], vert=False)
        plt.title('Боксплот количества возвратов за месяц')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = item_count_returns30.skew()
        kurtosis = item_count_returns30.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.item_count_returns30)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение количества фото в комментариеях по исходной выборке - 0, максимальное - 147. Размах значений составил 147. Рекомендуется дополнительно проанализировать товары с наибольшим количеством возвратов за месяц.
        2. Cреднее значение цены составляет примерно 0.073, а медианное - 0.0. Сдвиг незначительный.
        3. Рейтинг товаров до 90% не превышает 0, что свидетельствует о том, что у большинства товаров нет возвратов за месяц.
        4. На гистограмме видно, что распределение признака не похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии меньше 1, что не свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 70553.35 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def item_count_returns90_analitics(self, df_copy):
        item_count_returns90 = df_copy.item_count_returns90

        max_value = item_count_returns90.max()
        min_value = item_count_returns90.min()
        mean_value = item_count_returns90.mean()
        median_value = item_count_returns90.median()
        print(f'Наибольшее количество возвратов за квартал: {max_value}', f'Наименьшее количество возвратов за квартал: {min_value}',
              f'Среднее количество возвратов за квартал: {mean_value}', f'Медианное количество возвратов за квартал: {median_value}', sep='\n')

        percentile_10_value = item_count_returns90.quantile(0.10)
        percentile_25_value = item_count_returns90.quantile(0.25)
        percentile_50_value = item_count_returns90.quantile(0.50)
        percentile_75_value = item_count_returns90.quantile(0.75)
        percentile_90_value = item_count_returns90.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        item_count_returns90.describe()

        sns.histplot(item_count_returns90, bins=20, color='blue')
        plt.title('Гистограмма распределения количества возвратов за квартал')
        plt.xlabel('Количество возвратов за квартал')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.item_count_returns90 > percentile_90_value]))

        sns.histplot(item_count_returns90[item_count_returns90 > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения количества возвратов за квартал')
        plt.xlabel('Количество возвратов за квартал')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(item_count_returns90[item_count_returns90 > percentile_90_value], vert=False)
        plt.title('Боксплот количества возвратов за квартал')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = item_count_returns90.skew()
        kurtosis = item_count_returns90.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.item_count_returns90)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение количества фото в комментариеях по исходной выборке - 0, максимальное - 300. Размах значений составил 300. Рекомендуется дополнительно проанализировать товары с наибольшим количеством возвратов за квартал.
        2. Cреднее значение цены составляет примерно 0.183, а медианное - 0.0. Сдвиг незначительный.
        3. Рейтинг товаров до 90% не превышает 0, что свидетельствует о том, что у большинства товаров нет возвратов за квартал.
        4. На гистограмме видно, что распределение признака не похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии меньше 1, что не свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 69066.129 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def GmvTotal7_analitics(self, df_copy):
        GmvTotal7 = df_copy.GmvTotal7

        max_value = GmvTotal7.max()
        min_value = GmvTotal7.min()
        mean_value = GmvTotal7.mean()
        median_value = GmvTotal7.median()
        print(f'Наибольшая общая стоимость всех проданных товаров селлера за 7 дней: {max_value}', f'Наименьшая общая стоимость всех проданных товаров селлера за 7 дней: {min_value}',
              f'Средняя общая стоимость всех проданных товаров селлера за 7 дней: {mean_value}', f'Медианноая общая стоимость всех проданных товаров селлера за 7 дней: {median_value}', sep='\n')

        percentile_10_value = GmvTotal7.quantile(0.10)
        percentile_25_value = GmvTotal7.quantile(0.25)
        percentile_50_value = GmvTotal7.quantile(0.50)
        percentile_75_value = GmvTotal7.quantile(0.75)
        percentile_90_value = GmvTotal7.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        GmvTotal7.describe()

        sns.histplot(GmvTotal7, bins=20, color='blue')
        plt.title('Гистограмма распределения общей стоимости всех проданных товаров селлера за 7 дней')
        plt.xlabel('Общая стоимость всех проданных товаров селлера за 7 дней')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(GmvTotal7, vert=False)
        plt.title('Боксплот общей стоимости всех проданных товаров селлера за 7 дней')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = GmvTotal7.skew()
        kurtosis = GmvTotal7.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.GmvTotal7)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение количества фото в комментариеях по исходной выборке - 0, максимальное - 1998.058. Размах значений составил 1998.058. Рекомендуется дополнительно проанализировать товары с наибольшей общей стоимости всех проданных товаров селлера за 7 дней.
        2. Cреднее значение цены составляет примерно 1127.4062, а медианное - 1198.204. Сдвиг незначительный.
        3. 10-й процентиль: 870.6698714510658
        25-й процентиль: 1055.526389435682
        50-й процентиль: 1198.2046636612313
        75-й процентиль: 1303.1654256072668
        90-й процентиль: 1393.0610516051595
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии меньше 1, что не свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 13207.89 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def GmvTotal30_analitics(self, df_copy):
        GmvTotal30 = df_copy.GmvTotal30

        max_value = GmvTotal30.max()
        min_value = GmvTotal30.min()
        mean_value = GmvTotal30.mean()
        median_value = GmvTotal30.median()
        print(f'Наибольшая общая стоимость всех проданных товаров селлера за 30 дней: {max_value}', f'Наименьшая общая стоимость всех проданных товаров селлера за 30 дней: {min_value}',
              f'Средняя общая стоимость всех проданных товаров селлера за 30 дней: {mean_value}', f'Медианноая общая стоимость всех проданных товаров селлера за 30 дней: {median_value}', sep='\n')

        percentile_10_value = GmvTotal30.quantile(0.10)
        percentile_25_value = GmvTotal30.quantile(0.25)
        percentile_50_value = GmvTotal30.quantile(0.50)
        percentile_75_value = GmvTotal30.quantile(0.75)
        percentile_90_value = GmvTotal30.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        GmvTotal30.describe()

        sns.histplot(GmvTotal30, bins=20, color='blue')
        plt.title('Гистограмма распределения общей стоимости всех проданных товаров селлера за 30 дней')
        plt.xlabel('Общая стоимость всех проданных товаров селлера за 30 дней')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(GmvTotal30, vert=False)
        plt.title('Боксплот общей стоимости всех проданных товаров селлера за 30 дней')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = GmvTotal30.skew()
        kurtosis = GmvTotal30.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.GmvTotal30)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение количества фото в комментариеях по исходной выборке - 0, максимальное - 2137.83. Размах значений составил 12137.83. Рекомендуется дополнительно проанализировать товары с наибольшей общей стоимости всех проданных товаров селлера за 30 дней.
        2. Cреднее значение цены составляет примерно 1267.409, а медианное - 1343.55. Сдвиг незначительный.
        3. 10-й процентиль: 994.8778805185735
        25-й процентиль: 1198.7607216694548
        50-й процентиль: 1343.55082572947
        75-й процентиль: 1445.5655762420106
        90-й процентиль: 1537.8499607462675
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии меньше 1, что не свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 12816.13 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def GmvTotal90_analitics(self, df_copy):
        GmvTotal90 = df_copy.GmvTotal90

        max_value = GmvTotal90.max()
        min_value = GmvTotal90.min()
        mean_value = GmvTotal90.mean()
        median_value = GmvTotal90.median()
        print(f'Наибольшая общая стоимость всех проданных товаров селлера за 90 дней: {max_value}', f'Наименьшая общая стоимость всех проданных товаров селлера за 90 дней: {min_value}',
              f'Средняя общая стоимость всех проданных товаров селлера за 90 дней: {mean_value}', f'Медианноая общая стоимость всех проданных товаров селлера за 90 дней: {median_value}', sep='\n')

        percentile_10_value = GmvTotal90.quantile(0.10)
        percentile_25_value = GmvTotal90.quantile(0.25)
        percentile_50_value = GmvTotal90.quantile(0.50)
        percentile_75_value = GmvTotal90.quantile(0.75)
        percentile_90_value = GmvTotal90.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        GmvTotal90.describe()

        sns.histplot(GmvTotal90, bins=20, color='blue')
        plt.title('Гистограмма распределения общей стоимости всех проданных товаров селлера за 90 дней')
        plt.xlabel('Общая стоимость всех проданных товаров селлера за 90 дней')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(GmvTotal90, vert=False)
        plt.title('Боксплот общей стоимости всех проданных товаров селлера за 90 дней')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = GmvTotal90.skew()
        kurtosis = GmvTotal90.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.GmvTotal90)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение количества фото в комментариеях по исходной выборке - 0, максимальное - 2243.49. Размах значений составил 2243.49. Рекомендуется дополнительно проанализировать товары с наибольшей общей стоимости всех проданных товаров селлера за 90 дней.
        2. Cреднее значение цены составляет примерно 1363.405, а медианное - 1443.89. Сдвиг незначительный.
        3. 10-й процентиль: 1056.891836756863
        25-й процентиль: 1294.5317783768323
        50-й процентиль: 1443.893419082912
        75-й процентиль: 1553.6267733137054
        90-й процентиль: 1640.9649903156987
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии меньше 1, что не свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 13241.56 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def ExemplarAcceptedCountTotal7_analitics(self, df_copy):
        ExemplarAcceptedCountTotal7 = df_copy.ExemplarAcceptedCountTotal7

        max_value = ExemplarAcceptedCountTotal7.max()
        min_value = ExemplarAcceptedCountTotal7.min()
        mean_value = ExemplarAcceptedCountTotal7.mean()
        median_value = ExemplarAcceptedCountTotal7.median()
        print(f'Наибольшее общее количество товаров в заказах у селлера за 7 дней: {max_value}', f'Наименьшее общее количество товаров в заказах у селлера за 7 дней: {min_value}',
              f'Среднее общее количество товаров в заказах у селлера за 7 дней: {mean_value}', f'Медианное общее количество товаров в заказах у селлера за 7 дней: {median_value}', sep='\n')

        percentile_10_value = ExemplarAcceptedCountTotal7.quantile(0.10)
        percentile_25_value = ExemplarAcceptedCountTotal7.quantile(0.25)
        percentile_50_value = ExemplarAcceptedCountTotal7.quantile(0.50)
        percentile_75_value = ExemplarAcceptedCountTotal7.quantile(0.75)
        percentile_90_value = ExemplarAcceptedCountTotal7.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        ExemplarAcceptedCountTotal7.describe()

        sns.histplot(ExemplarAcceptedCountTotal7, bins=20, color='blue')
        plt.title('Гистограмма распределения общего количества товаров в заказах у селлера за 7 дней')
        plt.xlabel('Общее количество товаров в заказах у селлера за 7 дней')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.ExemplarAcceptedCountTotal7 > percentile_90_value]))

        sns.histplot(ExemplarAcceptedCountTotal7[ExemplarAcceptedCountTotal7 > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения общего количества товаров в заказах у селлера за 7 дней')
        plt.xlabel('Общее количество товаров в заказах у селлера за 7 дней')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(ExemplarAcceptedCountTotal7[ExemplarAcceptedCountTotal7 > percentile_90_value], vert=False)
        plt.title('Боксплот распределения общего количества товаров в заказах у селлера за 7 дней')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = ExemplarAcceptedCountTotal7.skew()
        kurtosis = ExemplarAcceptedCountTotal7.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.ExemplarAcceptedCountTotal7)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение количества фото в комментариеях по исходной выборке - 0, максимальное - 294526. Размах значений составил 294526. Рекомендуется дополнительно проанализировать товары с наибольшим общим количеством товаров в заказах у селлера за 7 дней.
        2. Cреднее значение цены составляет примерно 372.33, а медианное - 104. Сдвиг значительный.
        3. 10-й процентиль: 4.0
        25-й процентиль: 22.0
        50-й процентиль: 104.0
        75-й процентиль: 325.0
        90-й процентиль: 963.0
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии больше 1, что свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 38283.76 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def ExemplarAcceptedCountTotal30_analitics(self, df_copy):
        ExemplarAcceptedCountTotal30 = df_copy.ExemplarAcceptedCountTotal30

        max_value = ExemplarAcceptedCountTotal30.max()
        min_value = ExemplarAcceptedCountTotal30.min()
        mean_value = ExemplarAcceptedCountTotal30.mean()
        median_value = ExemplarAcceptedCountTotal30.median()
        print(f'Наибольшее общее количество товаров в заказах у селлера за 30 дней: {max_value}', f'Наименьшее общее количество товаров в заказах у селлера за 30 дней: {min_value}',
              f'Среднее общее количество товаров в заказах у селлера за 30 дней: {mean_value}', f'Медианное общее количество товаров в заказах у селлера за 30 дней: {median_value}', sep='\n')

        percentile_10_value = ExemplarAcceptedCountTotal30.quantile(0.10)
        percentile_25_value = ExemplarAcceptedCountTotal30.quantile(0.25)
        percentile_50_value = ExemplarAcceptedCountTotal30.quantile(0.50)
        percentile_75_value = ExemplarAcceptedCountTotal30.quantile(0.75)
        percentile_90_value = ExemplarAcceptedCountTotal30.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        ExemplarAcceptedCountTotal30.describe()

        sns.histplot(ExemplarAcceptedCountTotal30, bins=20, color='blue')
        plt.title('Гистограмма распределения общего количества товаров в заказах у селлера за 30 дней')
        plt.xlabel('Общее количество товаров в заказах у селлера за 30 дней')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.ExemplarAcceptedCountTotal30 > percentile_90_value]))

        sns.histplot(ExemplarAcceptedCountTotal30[ExemplarAcceptedCountTotal30 > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения общего количества товаров в заказах у селлера за 30 дней')
        plt.xlabel('Общее количество товаров в заказах у селлера за 30 дней')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(ExemplarAcceptedCountTotal30[ExemplarAcceptedCountTotal30 > percentile_90_value], vert=False)
        plt.title('Боксплот распределения общего количества товаров в заказах у селлера за 30 дней')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = ExemplarAcceptedCountTotal30.skew()
        kurtosis = ExemplarAcceptedCountTotal30.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.ExemplarAcceptedCountTotal30)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение количества фото в комментариеях по исходной выборке - 0, максимальное - 1350062. Размах значений составил 1350062. Рекомендуется дополнительно проанализировать товары с наибольшим общим количеством товаров в заказах у селлера за 30 дней.
        2. Cреднее значение цены составляет примерно 1497.799, а медианное - 430.0. Сдвиг значительный.
        3. 10-й процентиль: 11.0
        25-й процентиль: 87.0
        50-й процентиль: 430.0
        75-й процентиль: 1377.0
        90-й процентиль: 4238.0
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии больше 1, что свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 37880.5 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def ExemplarAcceptedCountTotal90_analitics(self, df_copy):
        ExemplarAcceptedCountTotal90 = df_copy.ExemplarAcceptedCountTotal90

        max_value = ExemplarAcceptedCountTotal90.max()
        min_value = ExemplarAcceptedCountTotal90.min()
        mean_value = ExemplarAcceptedCountTotal90.mean()
        median_value = ExemplarAcceptedCountTotal90.median()
        print(f'Наибольшее общее количество товаров в заказах у селлера за 90 дней: {max_value}', f'Наименьшее общее количество товаров в заказах у селлера за 90 дней: {min_value}',
              f'Среднее общее количество товаров в заказах у селлера за 90 дней: {mean_value}', f'Медианное общее количество товаров в заказах у селлера за 90 дней: {median_value}', sep='\n')

        percentile_10_value = ExemplarAcceptedCountTotal90.quantile(0.10)
        percentile_25_value = ExemplarAcceptedCountTotal90.quantile(0.25)
        percentile_50_value = ExemplarAcceptedCountTotal90.quantile(0.50)
        percentile_75_value = ExemplarAcceptedCountTotal90.quantile(0.75)
        percentile_90_value = ExemplarAcceptedCountTotal90.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        ExemplarAcceptedCountTotal90.describe()

        sns.histplot(ExemplarAcceptedCountTotal90, bins=20, color='blue')
        plt.title('Гистограмма распределения общего количества товаров в заказах у селлера за 90 дней')
        plt.xlabel('Общее количество товаров в заказах у селлера за 90 дней')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.ExemplarAcceptedCountTotal90 > percentile_90_value]))

        sns.histplot(ExemplarAcceptedCountTotal90[ExemplarAcceptedCountTotal90 > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения общего количества товаров в заказах у селлера за 90 дней')
        plt.xlabel('Общее количество товаров в заказах у селлера за 90 дней')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(ExemplarAcceptedCountTotal90[ExemplarAcceptedCountTotal90 > percentile_90_value], vert=False)
        plt.title('Боксплот распределения общего количества товаров в заказах у селлера за 90 дней')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = ExemplarAcceptedCountTotal90.skew()
        kurtosis = ExemplarAcceptedCountTotal90.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.ExemplarAcceptedCountTotal90)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение количества фото в комментариеях по исходной выборке - 0, максимальное - 4543683.0. Размах значений составил 4543683.0. Рекомендуется дополнительно проанализировать товары с наибольшим общим количеством товаров в заказах у селлера за 90 дней.
        2. Cреднее значение цены составляет примерно 4168.416, а медианное - 1207. Сдвиг значительный.
        3. 10-й процентиль: 22.0
        25-й процентиль: 233.0
        50-й процентиль: 1207.0
        75-й процентиль: 3961.0
        90-й процентиль: 10718.0
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии больше 1, что свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 38668.358 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def OrderAcceptedCountTotal7_analitics(self, df_copy):
        OrderAcceptedCountTotal7 = df_copy.OrderAcceptedCountTotal7

        max_value = OrderAcceptedCountTotal7.max()
        min_value = OrderAcceptedCountTotal7.min()
        mean_value = OrderAcceptedCountTotal7.mean()
        median_value = OrderAcceptedCountTotal7.median()
        print(f'Наибольшее общее количество принятых заказов за 7 дней: {max_value}', f'Наименьшее общее количество принятых заказов за 7 дней: {min_value}',
              f'Среднее общее количество принятых заказов за 7 дней: {mean_value}', f'Медианное общее количество принятых заказов за 7 дней: {median_value}', sep='\n')

        percentile_10_value = OrderAcceptedCountTotal7.quantile(0.10)
        percentile_25_value = OrderAcceptedCountTotal7.quantile(0.25)
        percentile_50_value = OrderAcceptedCountTotal7.quantile(0.50)
        percentile_75_value = OrderAcceptedCountTotal7.quantile(0.75)
        percentile_90_value = OrderAcceptedCountTotal7.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        OrderAcceptedCountTotal7.describe()

        sns.histplot(OrderAcceptedCountTotal7, bins=20, color='blue')
        plt.title('Гистограмма распределения общего количества принятых заказов за 7 дней')
        plt.xlabel('Общее количество принятых заказов за 7 дней')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.OrderAcceptedCountTotal7 > percentile_90_value]))

        sns.histplot(OrderAcceptedCountTotal7[OrderAcceptedCountTotal7 > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения общего количества принятых заказов за 7 дней')
        plt.xlabel('Общее количество принятых заказов за 7 дней')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(OrderAcceptedCountTotal7[OrderAcceptedCountTotal7 > percentile_90_value], vert=False)
        plt.title('Боксплот распределения общего количества принятых заказов за 7 дней')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = OrderAcceptedCountTotal7.skew()
        kurtosis = OrderAcceptedCountTotal7.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.OrderAcceptedCountTotal7)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение по исходной выборке - 0, максимальное - 229601.0. Размах значений составил 229601.0. Рекомендуется дополнительно проанализировать товары с наибольшим общим количеством принятых заказов за 7 дней.
        2. Cреднее значение составляет примерно 312.49, а медианное - 91. Сдвиг значительный.
        3. 10-й процентиль: 4.0
        25-й процентиль: 21.0
        50-й процентиль: 91.0
        75-й процентиль: 287.0
        90-й процентиль: 837.0
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии больше 1, что свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 36187.74 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def OrderAcceptedCountTotal30_analitics(self, df_copy):
        OrderAcceptedCountTotal30 = df_copy.OrderAcceptedCountTotal30

        max_value = OrderAcceptedCountTotal30.max()
        min_value = OrderAcceptedCountTotal30.min()
        mean_value = OrderAcceptedCountTotal30.mean()
        median_value = OrderAcceptedCountTotal30.median()
        print(f'Наибольшее общее количество принятых заказов за 30 дней: {max_value}', f'Наименьшее общее количество принятых заказов за 30 дней: {min_value}',
              f'Среднее общее количество принятых заказов за 30 дней: {mean_value}', f'Медианное общее количество принятых заказов за 30 дней: {median_value}', sep='\n')

        percentile_10_value = OrderAcceptedCountTotal30.quantile(0.10)
        percentile_25_value = OrderAcceptedCountTotal30.quantile(0.25)
        percentile_50_value = OrderAcceptedCountTotal30.quantile(0.50)
        percentile_75_value = OrderAcceptedCountTotal30.quantile(0.75)
        percentile_90_value = OrderAcceptedCountTotal30.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        OrderAcceptedCountTotal30.describe()

        sns.histplot(OrderAcceptedCountTotal30, bins=20, color='blue')
        plt.title('Гистограмма распределения общего количества принятых заказов за 30 дней')
        plt.xlabel('Общее количество принятых заказов за 30 дней')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.OrderAcceptedCountTotal30 > percentile_90_value]))

        sns.histplot(OrderAcceptedCountTotal30[OrderAcceptedCountTotal30 > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения общего количества принятых заказов за 30 дней')
        plt.xlabel('Общее количество принятых заказов за 30 дней')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(OrderAcceptedCountTotal30[OrderAcceptedCountTotal30 > percentile_90_value], vert=False)
        plt.title('Боксплот распределения общего количества принятых заказов за 30 дней')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = OrderAcceptedCountTotal30.skew()
        kurtosis = OrderAcceptedCountTotal30.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.OrderAcceptedCountTotal30)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение по исходной выборке - 0, максимальное - 1061315.0. Размах значений составил 1061315.0. Рекомендуется дополнительно проанализировать товары с наибольшим общим количеством принятых заказов за 30 дней.
        2. Cреднее значение составляет примерно 1270.1041, а медианное - 377.0. Сдвиг значительный.
        3. 10-й процентиль: 11.0
        25-й процентиль: 77.0
        50-й процентиль: 377.0
        75-й процентиль: 1171.0
        90-й процентиль: 3315.0
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии больше 1, что свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 36544.053 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def OrderAcceptedCountTotal90_analitics(self, df_copy):
        OrderAcceptedCountTotal90 = df_copy.OrderAcceptedCountTotal90

        max_value = OrderAcceptedCountTotal90.max()
        min_value = OrderAcceptedCountTotal90.min()
        mean_value = OrderAcceptedCountTotal90.mean()
        median_value = OrderAcceptedCountTotal90.median()
        print(f'Наибольшее общее количество принятых заказов за 90 дней: {max_value}', f'Наименьшее общее количество принятых заказов за 90 дней: {min_value}',
              f'Среднее общее количество принятых заказов за 90 дней: {mean_value}', f'Медианное общее количество принятых заказов за 90 дней: {median_value}', sep='\n')

        percentile_10_value = OrderAcceptedCountTotal90.quantile(0.10)
        percentile_25_value = OrderAcceptedCountTotal90.quantile(0.25)
        percentile_50_value = OrderAcceptedCountTotal90.quantile(0.50)
        percentile_75_value = OrderAcceptedCountTotal90.quantile(0.75)
        percentile_90_value = OrderAcceptedCountTotal90.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        OrderAcceptedCountTotal90.describe()

        sns.histplot(OrderAcceptedCountTotal90, bins=20, color='blue')
        plt.title('Гистограмма распределения общего количества принятых заказов за 90 дней')
        plt.xlabel('Общее количество принятых заказов за 90 дней')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.OrderAcceptedCountTotal90 > percentile_90_value]))

        sns.histplot(OrderAcceptedCountTotal90[OrderAcceptedCountTotal90 > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения общего количества принятых заказов за 90 дней')
        plt.xlabel('Общее количество принятых заказов за 90 дней')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(OrderAcceptedCountTotal90[OrderAcceptedCountTotal90 > percentile_90_value], vert=False)
        plt.title('Боксплот распределения общего количества принятых заказов за 90 дней')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = OrderAcceptedCountTotal90.skew()
        kurtosis = OrderAcceptedCountTotal90.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.OrderAcceptedCountTotal90)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение по исходной выборке - 0, максимальное - 3575912. Размах значений составил 3575912. Рекомендуется дополнительно проанализировать товары с наибольшим общим количеством принятых заказов за 90 дней.
        2. Cреднее значение составляет примерно 3637.395, а медианное - 1085 Сдвиг значительный.
        3. 10-й процентиль: 20.0
        25-й процентиль: 203.0
        50-й процентиль: 1085.0
        75-й процентиль: 3351.0
        90-й процентиль: 9631.0
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии больше 1, что свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 37826.43 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def ExemplarReturnedCountTotal7_analitics(self, df_copy):
        ExemplarReturnedCountTotal7 = df_copy.ExemplarReturnedCountTotal7

        max_value = ExemplarReturnedCountTotal7.max()
        min_value = ExemplarReturnedCountTotal7.min()
        mean_value = ExemplarReturnedCountTotal7.mean()
        median_value = ExemplarReturnedCountTotal7.median()
        print(f'Наибольшее общее количество возвращённых товаров селлера 7 дней: {max_value}', f'Наименьшее общее количество возвращённых товаров селлера 7 дней: {min_value}',
              f'Среднее общее количество возвращённых товаров селлера 7 дней: {mean_value}', f'Медианное общее количество возвращённых товаров селлера 7 дней: {median_value}', sep='\n')

        percentile_10_value = ExemplarReturnedCountTotal7.quantile(0.10)
        percentile_25_value = ExemplarReturnedCountTotal7.quantile(0.25)
        percentile_50_value = ExemplarReturnedCountTotal7.quantile(0.50)
        percentile_75_value = ExemplarReturnedCountTotal7.quantile(0.75)
        percentile_90_value = ExemplarReturnedCountTotal7.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        ExemplarReturnedCountTotal7.describe()

        sns.histplot(ExemplarReturnedCountTotal7, bins=20, color='blue')
        plt.title('Гистограмма распределения общего количества возвращённых товаров селлера 7 дней')
        plt.xlabel('Общее количество возвращённых товаров селлера 7 дней')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.OrderAcceptedCountTotal90 > percentile_90_value]))

        sns.histplot(ExemplarReturnedCountTotal7[ExemplarReturnedCountTotal7 > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения общего количества возвращённых товаров селлера 7 дней')
        plt.xlabel('Общее количество возвращённых товаров селлера 7 дней')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(ExemplarReturnedCountTotal7[ExemplarReturnedCountTotal7 > percentile_90_value], vert=False)
        plt.title('Боксплот распределения общего количества возвращённых товаров селлера 7 дней')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = ExemplarReturnedCountTotal7.skew()
        kurtosis = ExemplarReturnedCountTotal7.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.ExemplarReturnedCountTotal7)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение по исходной выборке - 0, максимальное - 12053. Размах значений составил 12053. Рекомендуется дополнительно проанализировать товары с наибольшим общим количеством возвращённых товаров селлера 7 дней.
        2. Cреднее значение составляет примерно 12.15204, а медианное - 3 Сдвиг значительный.
        3. 10-й процентиль: 0.0
        25-й процентиль: 0.0
        50-й процентиль: 3.0
        75-й процентиль: 12.0
        90-й процентиль: 31.0
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии больше 1, что свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 35904.71 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def ExemplarReturnedCountTotal30_analitics(self, df_copy):
        ExemplarReturnedCountTotal30 = df_copy.ExemplarReturnedCountTotal30

        max_value = ExemplarReturnedCountTotal30.max()
        min_value = ExemplarReturnedCountTotal30.min()
        mean_value = ExemplarReturnedCountTotal30.mean()
        median_value = ExemplarReturnedCountTotal30.median()
        print(f'Наибольшее общее количество возвращённых товаров селлера 30 дней: {max_value}', f'Наименьшее общее количество возвращённых товаров селлера 30 дней: {min_value}',
              f'Среднее общее количество возвращённых товаров селлера 30 дней: {mean_value}', f'Медианное общее количество возвращённых товаров селлера 30 дней: {median_value}', sep='\n')

        percentile_10_value = ExemplarReturnedCountTotal30.quantile(0.10)
        percentile_25_value = ExemplarReturnedCountTotal30.quantile(0.25)
        percentile_50_value = ExemplarReturnedCountTotal30.quantile(0.50)
        percentile_75_value = ExemplarReturnedCountTotal30.quantile(0.75)
        percentile_90_value = ExemplarReturnedCountTotal30.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        ExemplarReturnedCountTotal30.describe()

        sns.histplot(ExemplarReturnedCountTotal30, bins=20, color='blue')
        plt.title('Гистограмма распределения общего количества возвращённых товаров селлера 30 дней')
        plt.xlabel('Общее количество возвращённых товаров селлера 30 дней')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.ExemplarReturnedCountTotal30 > percentile_90_value]))

        sns.histplot(ExemplarReturnedCountTotal30[ExemplarReturnedCountTotal30 > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения общего количества возвращённых товаров селлера 30 дней')
        plt.xlabel('Общее количество возвращённых товаров селлера 30 дней')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(ExemplarReturnedCountTotal30[ExemplarReturnedCountTotal30 > percentile_90_value], vert=False)
        plt.title('Боксплот распределения общего количества возвращённых товаров селлера 30 дней')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = ExemplarReturnedCountTotal30.skew()
        kurtosis = ExemplarReturnedCountTotal30.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.ExemplarReturnedCountTotal30)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение по исходной выборке - 0, максимальное - 55681.0. Размах значений составил 55681.0. Рекомендуется дополнительно проанализировать товары с наибольшим общим количеством возвращённых товаров селлера 30 дней.
        2. Cреднее значение составляет примерно 49.71, а медианное - 14. Сдвиг значительный.
        3. 10-й процентиль: 0.0
        25-й процентиль: 2.0
        50-й процентиль: 14.0
        75-й процентиль: 49.0
        90-й процентиль: 128.0
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии больше 1, что свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 37364.14 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def ExemplarReturnedCountTotal90_analitics(self, df_copy):
        ExemplarReturnedCountTotal90 = df_copy.ExemplarReturnedCountTotal90

        max_value = ExemplarReturnedCountTotal90.max()
        min_value = ExemplarReturnedCountTotal90.min()
        mean_value = ExemplarReturnedCountTotal90.mean()
        median_value = ExemplarReturnedCountTotal90.median()
        print(f'Наибольшее общее количество возвращённых товаров селлера 90 дней: {max_value}', f'Наименьшее общее количество возвращённых товаров селлера 90 дней: {min_value}',
              f'Среднее общее количество возвращённых товаров селлера 90 дней: {mean_value}', f'Медианное общее количество возвращённых товаров селлера 90 дней: {median_value}', sep='\n')

        percentile_10_value = ExemplarReturnedCountTotal90.quantile(0.10)
        percentile_25_value = ExemplarReturnedCountTotal90.quantile(0.25)
        percentile_50_value = ExemplarReturnedCountTotal90.quantile(0.50)
        percentile_75_value = ExemplarReturnedCountTotal90.quantile(0.75)
        percentile_90_value = ExemplarReturnedCountTotal90.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        ExemplarReturnedCountTotal90.describe()

        sns.histplot(ExemplarReturnedCountTotal90, bins=20, color='blue')
        plt.title('Гистограмма распределения общего количества возвращённых товаров селлера 90 дней')
        plt.xlabel('Общее количество возвращённых товаров селлера 90 дней')
        plt.ylabel('Количество')
        plt.show()

        # количество объектов, значения цены по которым превышает 90 процентиль
        print(len(df_copy[df_copy.ExemplarReturnedCountTotal90 > percentile_90_value]))

        sns.histplot(ExemplarReturnedCountTotal90[ExemplarReturnedCountTotal90 > percentile_90_value], bins=20, color='blue')
        plt.title('Гистограмма распределения общего количества возвращённых товаров селлера 90 дней')
        plt.xlabel('Общее количество возвращённых товаров селлера 90 дней')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(ExemplarReturnedCountTotal90[ExemplarReturnedCountTotal90 > percentile_90_value], vert=False)
        plt.title('Боксплот распределения общего количества возвращённых товаров селлера 90 дней')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = ExemplarReturnedCountTotal90.skew()
        kurtosis = ExemplarReturnedCountTotal90.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.ExemplarReturnedCountTotal90)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение по исходной выборке - 0, максимальное - 181600. Размах значений составил 181600. Рекомендуется дополнительно проанализировать товары с наибольшим общим количеством возвращённых товаров селлера 90 дней.
        2. Cреднее значение составляет примерно 141.641, а медианное - 40. Сдвиг значительный.
        3. 10-й процентиль: 0.0
        25-й процентиль: 7.0
        50-й процентиль: 40.0
        75-й процентиль: 139.0
        90-й процентиль: 345.0
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии больше 1, что свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 39577.6 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def ExemplarReturnedValueTotal7_analitics(self, df_copy):
        ExemplarReturnedValueTotal7 = df_copy.ExemplarReturnedValueTotal7

        max_value = ExemplarReturnedValueTotal7.max()
        min_value = ExemplarReturnedValueTotal7.min()
        mean_value = ExemplarReturnedValueTotal7.mean()
        median_value = ExemplarReturnedValueTotal7.median()
        print(f'Наибольшая суммарная стоимость всех возвращённых товаров за последние 7 дней: {max_value}', f'Наименьшая суммарная стоимость всех возвращённых товаров за последние 7 дней: {min_value}',
              f'Средняя суммарная стоимость всех возвращённых товаров за последние 7 дней: {mean_value}', f'Медианная суммарная стоимость всех возвращённых товаров за последние 7 дней: {median_value}', sep='\n')

        percentile_10_value = ExemplarReturnedValueTotal7.quantile(0.10)
        percentile_25_value = ExemplarReturnedValueTotal7.quantile(0.25)
        percentile_50_value = ExemplarReturnedValueTotal7.quantile(0.50)
        percentile_75_value = ExemplarReturnedValueTotal7.quantile(0.75)
        percentile_90_value = ExemplarReturnedValueTotal7.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        ExemplarReturnedValueTotal7.describe()

        sns.histplot(ExemplarReturnedValueTotal7, bins=20, color='blue')
        plt.title('Гистограмма распределения суммарной стоимости всех возвращённых товаров за последние 7 дней')
        plt.xlabel('Суммарная стоимость всех возвращённых товаров за последние 7 дней')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(ExemplarReturnedValueTotal7, vert=False)
        plt.title('Боксплот распределения суммарной стоимости всех возвращённых товаров за последние 7 дней')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = ExemplarReturnedValueTotal7.skew()
        kurtosis = ExemplarReturnedValueTotal7.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.ExemplarReturnedValueTotal7)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение по исходной выборке - 0, максимальное - 1663.665. Размах значений составил 1663.665. Рекомендуется дополнительно проанализировать товары с наибольшей суммарной стоимостью всех возвращённых товаров за последние 7 дней'.
        2. Cреднее значение составляет примерно 665.26, а медианное - 839.7. Сдвиг значительный.
        3. 10-й процентиль: 0.0
        25-й процентиль: 0.0
        50-й процентиль: 839.7457204175441
        75-й процентиль: 984.1122748879271
        90-й процентиль: 1095.773736613491
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии меньше 1, что свидетельствует о нескошенности вправо. Коэффициент эксцесса меньше 1, что говорит о рассредоточенности значений от средних. По проведенному тесту на нормальность распределения 17282.679 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def ExemplarReturnedValueTotal30_analitics(self, df_copy):
        ExemplarReturnedValueTotal30 = df_copy.ExemplarReturnedValueTotal30

        max_value = ExemplarReturnedValueTotal30.max()
        min_value = ExemplarReturnedValueTotal30.min()
        mean_value = ExemplarReturnedValueTotal30.mean()
        median_value = ExemplarReturnedValueTotal30.median()
        print(f'Наибольшая суммарная стоимость всех возвращённых товаров за последние 30 дней: {max_value}', f'Наименьшая суммарная стоимость всех возвращённых товаров за последние 30 дней: {min_value}',
              f'Средняя суммарная стоимость всех возвращённых товаров за последние 30 дней: {mean_value}', f'Медианная суммарная стоимость всех возвращённых товаров за последние 30 дней: {median_value}', sep='\n')

        percentile_10_value = ExemplarReturnedValueTotal30.quantile(0.10)
        percentile_25_value = ExemplarReturnedValueTotal30.quantile(0.25)
        percentile_50_value = ExemplarReturnedValueTotal30.quantile(0.50)
        percentile_75_value = ExemplarReturnedValueTotal30.quantile(0.75)
        percentile_90_value = ExemplarReturnedValueTotal30.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        ExemplarReturnedValueTotal30.describe()

        sns.histplot(ExemplarReturnedValueTotal30, bins=20, color='blue')
        plt.title('Гистограмма распределения суммарной стоимости всех возвращённых товаров за последние 30 дней')
        plt.xlabel('Суммарная стоимость всех возвращённых товаров за последние 30 дней')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(ExemplarReturnedValueTotal30, vert=False)
        plt.title('Боксплот распределения суммарной стоимости всех возвращённых товаров за последние 30 дней')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = ExemplarReturnedValueTotal30.skew()
        kurtosis = ExemplarReturnedValueTotal30.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.ExemplarReturnedValueTotal30)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение по исходной выборке - 0, максимальное - 1807.58. Размах значений составил 1807.58. Рекомендуется дополнительно проанализировать товары с наибольшей суммарной стоимостью всех возвращённых товаров за последние 30 дней.
        2. Cреднее значение составляет примерно 869.31, а медианное - 997.7887. Сдвиг значительный.
        3. 10-й процентиль: 0.0
        25-й процентиль: 824.433621604529
        50-й процентиль: 997.7887489436037
        75-й процентиль: 1125.810305634168
        90-й процентиль: 1233.2359292794854
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии меньше 1, что свидетельствует о нескошенности вправо. Коэффициент эксцесса меньще 1, что говорит о рассредоточенности значений от средних. По проведенному тесту на нормальность распределения 17618.77 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def ExemplarReturnedValueTotal90_analitics(self, df_copy):
        ExemplarReturnedValueTotal90 = df_copy.ExemplarReturnedValueTotal90

        max_value = ExemplarReturnedValueTotal90.max()
        min_value = ExemplarReturnedValueTotal90.min()
        mean_value = ExemplarReturnedValueTotal90.mean()
        median_value = ExemplarReturnedValueTotal90.median()
        print(f'Наибольшая суммарная стоимость всех возвращённых товаров за последние 90 дней: {max_value}', f'Наименьшая суммарная стоимость всех возвращённых товаров за последние 90 дней: {min_value}',
              f'Средняя суммарная стоимость всех возвращённых товаров за последние 90 дней: {mean_value}', f'Медианная суммарная стоимость всех возвращённых товаров за последние 90 дней: {median_value}', sep='\n')

        percentile_10_value = ExemplarReturnedValueTotal90.quantile(0.10)
        percentile_25_value = ExemplarReturnedValueTotal90.quantile(0.25)
        percentile_50_value = ExemplarReturnedValueTotal90.quantile(0.50)
        percentile_75_value = ExemplarReturnedValueTotal90.quantile(0.75)
        percentile_90_value = ExemplarReturnedValueTotal90.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        ExemplarReturnedValueTotal90.describe()

        sns.histplot(ExemplarReturnedValueTotal90, bins=20, color='blue')
        plt.title('Гистограмма распределения суммарной стоимости всех возвращённых товаров за последние 90 дней')
        plt.xlabel('Суммарная стоимость всех возвращённых товаров за последние 90 дней')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(ExemplarReturnedValueTotal90, vert=False)
        plt.title('Боксплот распределения суммарной стоимости всех возвращённых товаров за последние 90 дней')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = ExemplarReturnedValueTotal90.skew()
        kurtosis = ExemplarReturnedValueTotal90.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.ExemplarReturnedValueTotal90)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение по исходной выборке - 0, максимальное - 1914.00028. Размах значений составил 1914.00028. Рекомендуется дополнительно проанализировать товары с наибольшей суммарной стоимостью всех возвращённых товаров за последние 90 дней.
        2. Cреднее значение составляет примерно 986.60658, а медианное - 1107.36. Сдвиг значительный.
        3. 10-й процентиль: 0.0
        25-й процентиль: 938.5802260007381
        50-й процентиль: 1107.3675046140497
        75-й процентиль: 1225.0433131943137
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии меньше 1, что свидетельствует о нескошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 18020.2 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def ItemVarietyCount_analitics(self, df_copy):
        ItemVarietyCount = df_copy.ItemVarietyCount

        max_value = ItemVarietyCount.max()
        min_value = ItemVarietyCount.min()
        mean_value = ItemVarietyCount.mean()
        median_value = ItemVarietyCount.median()
        print(f'Наибольшее количество разновидностей одного товара представлено у продавца: {max_value}', f'Наименьшее количество разновидностей одного товара представлено у продавца: {min_value}',
              f'Среднее количество разновидностей одного товара представлено у продавца: {mean_value}', f'Медианное количество разновидностей одного товара представлено у продавца: {median_value}', sep='\n')

        percentile_10_value = ItemVarietyCount.quantile(0.10)
        percentile_25_value = ItemVarietyCount.quantile(0.25)
        percentile_50_value = ItemVarietyCount.quantile(0.50)
        percentile_75_value = ItemVarietyCount.quantile(0.75)
        percentile_90_value = ItemVarietyCount.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        ItemVarietyCount.describe()

        sns.histplot(ItemVarietyCount, bins=20, color='blue')
        plt.title('Гистограмма распределения количества разновидностей одного товара представлено у продавца')
        plt.xlabel('Количество разновидностей одного товара представлено у продавца')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(ItemVarietyCount, vert=False)
        plt.title('Боксплот распределения количества разновидностей одного товара представлено у продавца')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = ItemVarietyCount.skew()
        kurtosis = ItemVarietyCount.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.ItemVarietyCount)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение по исходной выборке - 0, максимальное - 1548.0. Размах значений составил 1548.0. Рекомендуется дополнительно проанализировать товары с наибольшим количеством разновидностей одного товара представлено у продавца.
        2. Cреднее значение составляет примерно 98.291, а медианное - 7. Сдвиг значительный.
        3. 10-й процентиль: 1.0
        25-й процентиль: 2.0
        50-й процентиль: 7.0
        75-й процентиль: 58.0
        90-й процентиль: 307.0
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии больше 1, что свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 40143.63> [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def ItemAvailableCount_analitics(self, df_copy):
        ItemAvailableCount = df_copy.ItemAvailableCount

        max_value = ItemAvailableCount.max()
        min_value = ItemAvailableCount.min()
        mean_value = ItemAvailableCount.mean()
        median_value = ItemAvailableCount.median()
        print(f'Наибольшее количество доступных товаров представлено у продавца: {max_value}', f'Наименьшее количество доступных товаров представлено у продавца: {min_value}',
              f'Среднее количество доступных товаров представлено у продавца: {mean_value}', f'Медианное количество доступных товаров представлено у продавца: {median_value}', sep='\n')

        percentile_10_value = ItemAvailableCount.quantile(0.10)
        percentile_25_value = ItemAvailableCount.quantile(0.25)
        percentile_50_value = ItemAvailableCount.quantile(0.50)
        percentile_75_value = ItemAvailableCount.quantile(0.75)
        percentile_90_value = ItemAvailableCount.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        ItemAvailableCount.describe()

        sns.histplot(ItemAvailableCount, bins=20, color='blue')
        plt.title('Гистограмма распределения количества доступных товаров представлено у продавца')
        plt.xlabel('Количество доступных товаров представлено у продавца')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(ItemAvailableCount, vert=False)
        plt.title('Боксплот распределения количества доступных товаров представлено у продавца')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = ItemAvailableCount.skew()
        kurtosis = ItemAvailableCount.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.ItemAvailableCount)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение по исходной выборке - 0, максимальное - 1548.0. Размах значений составил 1548.0. Рекомендуется дополнительно проанализировать товары с наибольшим количеством доступных товаров представлено у продавца.
        2. Cреднее значение составляет примерно 98.233, а медианное - 7. Сдвиг значительный.
        3. 10-й процентиль: 1.0
        25-й процентиль: 2.0
        50-й процентиль: 7.0
        75-й процентиль: 56.0
        90-й процентиль: 307.0
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии больше 1, что свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 40158.99 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def seller_time_alive_analitics(self, df_copy):
        seller_time_alive = df_copy.seller_time_alive

        max_value = seller_time_alive.max()
        min_value = seller_time_alive.min()
        mean_value = seller_time_alive.mean()
        median_value = seller_time_alive.median()
        print(f'Наибольшее количество времени продавец активен на платформе: {max_value}', f'Наименьшее количество времени продавец активен на платформе: {min_value}',
              f'Среднее количество времени продавец активен на платформе: {mean_value}', f'Медианное количество времени продавец активен на платформе: {median_value}', sep='\n')

        percentile_10_value = seller_time_alive.quantile(0.10)
        percentile_25_value = seller_time_alive.quantile(0.25)
        percentile_50_value = seller_time_alive.quantile(0.50)
        percentile_75_value = seller_time_alive.quantile(0.75)
        percentile_90_value = seller_time_alive.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        seller_time_alive.describe()

        sns.histplot(seller_time_alive, bins=20, color='blue')
        plt.title('Гистограмма распределения количества времени продавец активен на платформе')
        plt.xlabel('Количество времени продавец активен на платформеа')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(seller_time_alive, vert=False)
        plt.title('Боксплот распределения количества времени продавец активен на платформе')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = seller_time_alive.skew()
        kurtosis = seller_time_alive.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.seller_time_alive)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение по исходной выборке - 1.0, максимальное - 2265.0. Размах значений составил 2264.0. Рекомендуется дополнительно проанализировать товары с наибольшим количеством времени продавец активен на платформе.
        2. Cреднее значение составляет примерно 684.57, а медианное - 607. Сдвиг незначительный.
        3. 10-й процентиль: 78.0
        25-й процентиль: 262.0
        50-й процентиль: 607.0
        75-й процентиль: 1058.0
        90-й процентиль: 1353.0
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии меньше 1, что свидетельствует о нескошенности вправо. Коэффициент эксцесса меньше 1, что говорит о рассредоточенности значений от средних. По проведенному тесту на нормальность распределения 2489.915 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    def item_time_alive_analitics(self, df_copy):
        item_time_alive = df_copy.item_time_alive

        max_value = item_time_alive.max()
        min_value = item_time_alive.min()
        mean_value = item_time_alive.mean()
        median_value = item_time_alive.median()
        print(f'Наибольшее количество времени товар находится в продаже до времени проверки: {max_value}', f'Наименьшее количество времени товар находится в продаже до времени проверкие: {min_value}',
              f'Среднее количество времени товар находится в продаже до времени проверки: {mean_value}', f'Медианное количество времени товар находится в продаже до времени проверки: {median_value}', sep='\n')

        percentile_10_value = item_time_alive.quantile(0.10)
        percentile_25_value = item_time_alive.quantile(0.25)
        percentile_50_value = item_time_alive.quantile(0.50)
        percentile_75_value = item_time_alive.quantile(0.75)
        percentile_90_value = item_time_alive.quantile(0.90)
        print(f'10-й процентиль: {percentile_10_value}',
              f'25-й процентиль: {percentile_25_value}',
              f'50-й процентиль: {percentile_50_value}',
              f'75-й процентиль: {percentile_75_value}',
              f'90-й процентиль: {percentile_90_value}', sep='\n')

        item_time_alive.describe()

        sns.histplot(item_time_alive, bins=20, color='blue')
        plt.title('Гистограмма распределения количества времени товар находится в продаже до времени проверки')
        plt.xlabel('Количество времени товар находится в продаже до времени проверки')
        plt.ylabel('Количество')
        plt.show()

        plt.boxplot(item_time_alive, vert=False)
        plt.title('Боксплот распределения количества времени товар находится в продаже до времени проверки')
        plt.show()

        # коэффициенты ассиметрии и эксцесса
        skew = item_time_alive.skew()
        kurtosis = item_time_alive.kurtosis()
        print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

        # тест на нормальность распределения
        # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
        # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
        # Уровень значимости полагаем равным 0,05.
        # Проверим признаки на нормальность при помощи  теста Андерсона-Дарлинга(более 5000):
        result = stats.anderson(df_copy.item_time_alive)
        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")
        print(f"Significance levels: {result.significance_level}")

        """**Выводы:**
        1. Минимальное значение по исходной выборке - 0, максимальное - 2185.0. Размах значений составил 2185.0. Рекомендуется дополнительно проанализировать товары с наибольшим количеством времени товар находится в продаже до времени проверки.
        2. Cреднее значение составляет примерно 274.667, а медианное - 126. Сдвиг значительный.
        3. 10-й процентиль: 2.0
        25-й процентиль: 7.0
        50-й процентиль: 126.0
        75-й процентиль: 444.0
        90-й процентиль: 797.0
        4. На гистограмме видно, что распределение признака похоже на нормальное. Боксплот показывает наличие оставшихся выбросов.
        5. Коэффициент ассиметрии больше 1, что свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения 13582.2 > [0.576 0.656 0.787 0.918 1.092], поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
        """

    """
    ###Категориальные признаки (resolution, brand_name, SellerID, CommercialCategory)
    
    ####resolution
    """

    def resolution_analitics(self, df_copy):

        resolution = df_copy.resolution
        print(resolution.describe())

        print(resolution.mode())

        print(resolution.value_counts())

        top_categories = resolution.value_counts().nlargest(10)
        plt.pie(top_categories, labels=top_categories.index, autopct='%.1f')
        plt.title('Распределение по фейковости товаров')
        plt.show()

        """**Выводы:**
        1. Количество уникальных значений составляет 2.
        2. Самая часто встречающаяся - 0 (93.4%).
        """

    """
    ####brand_name
    """
    def brand_name_analitics(self, df_copy):
        brand_name = df_copy.brand_name
        print(brand_name.describe())

        print(brand_name.mode())

        print(brand_name.value_counts())

        top_categories = brand_name.value_counts().nlargest(10)
        plt.pie(top_categories, labels=top_categories.index, autopct='%.1f')
        plt.title('Распределение по основным брендам товаров')
        plt.show()

        """**Выводы:**
        1. Количество уникальных значений составляет 4067.
        2. Самая часто встречающийся бренд - iQZiP (5.4%).
        3. Основную массу на рынке составляют iQZiP, ProFDetali, OEM.
        """

    """
    ####SellerID
    """
    def SellerID_analitics(self, df_copy):
        SellerID = df_copy.SellerID
        print(SellerID.describe())

        print(SellerID.mode())

        print(SellerID.value_counts())

        filtered_supplierId = SellerID[SellerID != 0]
        top_categories = filtered_supplierId.value_counts().nlargest(10)
        plt.pie(top_categories, labels=top_categories.index, autopct='%.1f')
        plt.title('Распределение по продавцам товаров')
        plt.show()

        """**Выводы:**
        1. Самая часто встречающийся id - 24 (23%).
        2. Основную массу на рынке составляют 24, 69, 442.
        """
    """
    ####CommercialCategory
    """
    def CommercialCategory_analitics(self, df_copy):
        CommercialCategory = df_copy.CommercialCategory
        print(CommercialCategory.describe())

        print(CommercialCategory.mode())

        print(CommercialCategory.value_counts())

        top_categories = CommercialCategory.value_counts().nlargest(10)
        plt.pie(top_categories, labels=top_categories.index, autopct='%.1f')
        plt.title('Распределение по id продавцов товаров')
        plt.show()

        """**Выводы:**
        1. Самая часто встречающаяся - Дисплеи для телефонов (17.2%).
        2. Основную массу на рынке составляют Дисплеи для телефонов, Аккумулятор для мобильного телефона, Корпуса для телефонов.
        """

    """
    TF-IDF / FastText эмбединги + кластеризация
    Ты используешь FastText — логично, ведь он работает хорошо с морфологией. Вопрос: а что ты с ним делаешь дальше?
    
    Рекомендации:
    Можно кластеризовать текстовые описания с помощью KMeans или HDBSCAN → получить признак text_cluster_id
    Можно добавить расстояние до "эталонных" описаний известных брендов
    Можно обучить simple классификатор (LogReg) на текстах — отдельно от всех табличных фичей — и использовать его логит как фичу
    """
    """
    Заголовок
    """
    def name_fraud_detection(self, df_copy):
        processor = Text_Preprocessing()

        # нормализация, токенизация, лемматизация и векторизация
        print('нормализация, токенизация, лемматизация и векторизация name')
        empty_names = df_copy[df_copy['name_rus'].str.strip() == '']
        print(f"Пустых названий: {len(empty_names)}")
        print(empty_names.head())
        df_copy.loc[:,'name_rus'] = df_copy['name_rus'].replace('', np.nan)
        df_copy = df_copy.dropna(subset=['name_rus'])
        print(f"После очистки: {df_copy.shape[0]} строк")
        sample_df = df_copy[:100].copy()

        name_clean = []
        name_vecs = []
        print('нормализация, токенизация, лемматизация и векторизация name')
        for string in tqdm(sample_df['name_rus'], desc="Обработка name_rus"):
            if not isinstance(string, str) or string.strip() == '':
                name_clean.append([])
                name_vecs.append(np.zeros(300))  # если размерность TF-IDF вектора 300, или как у тебя
                continue
            normalized_string = processor.Normalization(string, processor.stop_words)
            lemmatized_string = processor.Tokenization_Lemmatization(normalized_string)
            name_clean.append(lemmatized_string)
            if not lemmatized_string:
                vectorized_string = np.zeros(300)
            else:
                vectorized_string = processor.Embedding_Tfidf(lemmatized_string)
            name_vecs.append(vectorized_string)
        sample_df.loc[:, 'name_clean'] = name_clean
        sample_df.loc[:, 'name_vec'] = name_vecs
        print("Сформироавно name_clean name_vec")

        """
        Кластеризация векторных представлений
        
        После получения name_vec можно применить кластеризацию (KMeans, DBSCAN, HDBSCAN).
        Кластеры с редкими или аномальными текстами можно пометить как подозрительные.
        Это поможет находить шаблонные фродовые названия.
        
        """

        X = np.vstack(sample_df['name_vec'].values)

        # масштибируем фичи
        X_norm = normalize(X)

        # обучаем DBSCAN
        hdbscan = HDBSCAN(min_cluster_size=5, metric='euclidean')
        clusters = hdbscan.fit_predict(X_norm)

        """
        Добавить кластер ID как признак
        """
        sample_df.loc[:, 'name_cluster'] = clusters
        sample_df.loc[:, 'is_outlier_name'] = sample_df['name_cluster'] == -1

        print("Подозрительные названия (кластеры -1):")
        print(sample_df[sample_df['is_outlier_name']][['name_rus', 'name_cluster']].head(20))

        sample_df = sample_df.copy()
        return sample_df

    def max_brand_similarity(self, name_vec, brand_vecs):
        """
        Вычисляет максимальное косинус  ное сходство между вектором названия товара
        и всеми векторами брендов.
        """
        best_brand = None
        best_score = 0
        for brand, brand_vec in brand_vecs.items():
            if np.linalg.norm(name_vec) == 0 or np.linalg.norm(brand_vec) == 0:
                sim = 0
            else:
                sim = cosine_similarity(name_vec.reshape(1, -1), brand_vec.reshape(1, -1))[0][0]
            if sim > best_score:
                best_score = sim
                best_brand = brand
        return best_brand, best_score

    def brand_fraud_detection(self, df_copy):
        processor = Text_Preprocessing()
        df_copy = df_copy.copy()

        # нормализация, токенизация, лемматизация и векторизация
        print('нормализация, токенизация, лемматизация и векторизация brand_name')
        empty_brands = df_copy[df_copy['brand_name'].str.strip() == '']
        print(f"Пустых брендов: {len(empty_brands)}")
        print(empty_brands.head())
        df_copy.loc[:, 'brand_name'] = df_copy['brand_name'].replace('', np.nan)
        df_copy = df_copy.dropna(subset=['brand_name'])
        print(f"После очистки: {df_copy.shape[0]} строк")

        brand_clean = []
        brand_vecs = []
        brand_vectors = {}
        print('нормализация, токенизация, лемматизация и векторизация brand_name')
        for string in tqdm(df_copy['brand_name'], desc="Обработка brand_name"):
            if not isinstance(string, str) or string.strip() == '':
                brand_clean.append([])
                brand_vecs.append(np.zeros(300))  # если размерность TF-IDF вектора 300, или как у тебя
                continue
            normalized_string = processor.Normalization(string, processor.stop_words)
            lemmatized_string = processor.Tokenization_Lemmatization(normalized_string)
            brand_clean.append(lemmatized_string)
            if not lemmatized_string:
                vectorized_string = np.zeros(300)
            else:
                vectorized_string = processor.Embedding_Tfidf(lemmatized_string)
            brand_vecs.append(vectorized_string)
            brand_vectors[string] = vectorized_string

        df_copy['brand_clean'] = brand_clean
        df_copy['brand_vecs'] = brand_vecs


        print("Сформироавно brand_clean brand_vecs")
        """
        Список эталонных брендов (можно вручную или из базы).
        Получить вектор для каждого бренда (FastText). 
        Для каждого названия в данных:
            получить его вектор,
            посчитать косинусное сходство с каждым брендом,
            выбрать максимум (или топ-N) — и сохранить как признак.
        """
        # 1. Убедись, что это явная копия
        df_copy = df_copy.copy()

        # 2. Получи Series из apply, распакуй в отдельный DataFrame
        brand_results = df_copy['name_vec'].apply(
            lambda vec: pd.Series(self.max_brand_similarity(vec, brand_vectors))
        )
        brand_results.columns = ['matched_brand', 'brand_similarity_max']

        # 3. Добавь их в df_copy безопасно
        df_copy['matched_brand'] = brand_results['matched_brand']
        df_copy['brand_similarity_max'] = brand_results['brand_similarity_max']

        df_copy = df_copy.copy()
        return df_copy

    """
    Кластеризация description (если есть колонка с описанием товара) → desc_cluster
    Визуализация кластеров (UMAP, t-SNE) — чтобы наглядно видеть фейковые группы
    Размер кластера как признак (cluster_size)
    Расстояние до центра кластера → подозрительные точки — на краях
    """
    """
    Функция для нахождения центров кластеров
    """
    def get_cluster_centers(self, X_norm, labels):
        centers = {}
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            cluster_points = X_norm[labels == cluster_id]
            centers[cluster_id] = cluster_points.mean(axis=0)
        return centers

    def description_fraud_detection(self, df_copy):
        processor = Text_Preprocessing()
        df_copy = df_copy.copy()
        # нормализация, токенизация, лемматизация и векторизация
        print('нормализация, токенизация, лемматизация и векторизация description')
        empty_descriptions = df_copy[df_copy['description'].str.strip() == '']
        print(f"Пустых описаний: {len(empty_descriptions)}")
        print(empty_descriptions.head())
        df_copy.loc[:, 'description'] = df_copy['description'].replace('', np.nan)
        df_copy = df_copy.dropna(subset=['description'])
        print(f"После очистки: {df_copy.shape[0]} строк")

        description_clean = []
        description_vecs = []
        print('нормализация, токенизация, лемматизация и векторизация description')
        for string in tqdm(df_copy['description'], desc="Обработка description"):
            if not isinstance(string, str) or string.strip() == '':
                description_clean.append([])
                description_vecs.append(np.zeros(300))  # если размерность TF-IDF вектора 300, или как у тебя
                continue
            normalized_string = processor.Normalization(string, processor.stop_words)
            lemmatized_string = processor.Tokenization_Lemmatization(normalized_string)
            description_clean.append(lemmatized_string)
            if not lemmatized_string:
                vectorized_string = np.zeros(300)
            else:
                vectorized_string = processor.Embedding_Tfidf(lemmatized_string)
            description_vecs.append(vectorized_string)
        df_copy.loc[:, 'description_clean'] = description_clean
        df_copy.loc[:, 'description_vecs'] = description_vecs
        print("Сформироавно description_clean description_vecs")

        """
        Кластеризация векторных представлений
    
        После получения name_vec можно применить кластеризацию (KMeans, DBSCAN, HDBSCAN).
        Кластеры с редкими или аномальными текстами можно пометить как подозрительные.
        Это поможет находить шаблонные фродовые описания.
    
        """

        X = np.vstack(df_copy['description_vecs'].values)

        # масштибируем фичи
        X_norm = normalize(X)

        # обучаем DBSCAN
        hdbscan = HDBSCAN(min_cluster_size=5, metric='euclidean')
        clusters = hdbscan.fit_predict(X_norm)

        """
        Добавить кластер ID как признак
        """
        df_copy.loc[:, 'description_cluster'] = clusters
        df_copy.loc[:, 'is_outlier_description'] = df_copy['description_cluster'] == -1
        cluster_counts = df_copy['description_cluster'].value_counts().to_dict()
        df_copy['description_cluster_size'] = df_copy['description_cluster'].map(cluster_counts)

        print("Подозрительные названия (кластеры -1):")
        print(df_copy[df_copy['is_outlier_description']][['description', 'description_cluster']].head(20))

        cluster_centers = self.get_cluster_centers(X_norm, clusters)
        distances = []
        for vec, cluster_id in zip(X_norm, clusters):
            if cluster_id == -1:
                distances.append(-1)
            else:
                center = cluster_centers[cluster_id]
                dist = np.linalg.norm(vec-center)
                distances.append(dist)
        df_copy.loc[:, 'description_distance_to_center'] = distances
        df_copy.loc[:, 'log_description_distance_to_center'] = np.log1p(df_copy['description_distance_to_center'].clip(lower=0).fillna(0))
        """
        Визуализация кластеров (если ты хочешь её реально использовать)
        Сюда можно добавить UMAP / t-SNE
        """
        embedding = UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean').fit_transform(X_norm)
        umap_1 = embedding[:,0]
        df_copy.loc[:, 'umap_1'] = umap_1
        umap_2 = embedding[:, 1]
        df_copy.loc[:, 'umap_2'] = umap_2

        df_vis = df_copy[~df_copy['umap_1'].isna() & ~df_copy['umap_2'].isna()]
        plt.figure(figsize=(10, 6))
        plt.scatter(df_vis['umap_1'], df_vis['umap_2'], c=df_vis['description_cluster'], cmap='Spectral', s=5)
        plt.title('UMAP projection of description clusters')
        plt.show()

        df_copy.loc[:, 'description_length'] = df_copy['description'].apply(len)

        df_copy = df_copy.copy()
        return df_copy

    """
    Что точно можно и нужно подавать в CatBoost:
    
    Все табличные фичи:
    Фичи из feature_engineering_* (returns, ratings, sales, seller, price и т.д.)
    Аггрегации на уровне бренда и продавца
    Кластеры и расстояния до центра кластера (*_cluster, *_distance_to_center, *_cluster_size)
    Логиты из LogisticRegression как фичи (logits_description)
    Категориальные фичи (если они есть):
    
    У CatBoost есть нативная поддержка категориальных переменных (строковые фичи, такие как CommercialCategory, brand_name, SellerID, seller_age_bucket)
    Их можно подавать как есть в список cat_features=[]
    
    ⚠️ Что нужно преобразовать перед подачей:
    
    Списки / эмбеддинги как np.array / list (например name_vec, description_vecs) — подавать нельзя напрямую.
    
    Лучше использовать их производные:
    Кластеры (*_cluster)
    Расстояния до центра (*_distance_to_center)
    Симиларити (brand_similarity_max)
    Логиты из ML-моделей (logits_description)
    
    👉 Подавать сами вектора нельзя — CatBoost их не воспримет корректно (он не умеет работать с np.array в качестве отдельных фичей).
    
    Фичи с объектами / списками (name_clean, brand_clean, description_clean) — тоже исключить.
    Их роль уже учтена в векторах → которые преобразованы в полезные числовые признаки выше.
    
    🚫 Что не подавать:
    Сырые текстовые признаки (name_rus, brand_name, description) — если ты уже превратил их в числовые признаки, они не нужны.
    Сами эмбеддинги (*_vec) как np.array / list
    Любые фичи с типом object, кроме категориальных
    """
    def train_model(self, df_copy):
        """
        Обучение CatBoost и PCA + сохранение
        """
        """
        Снизить размерность эмбеддингов
        
        Используй алгоритмы понижения размерности, например:
        PCA (главные компоненты)
        UMAP
        t-SNE (для визуализации, но не всегда подходит для признаков)
        Autoencoder
        PCA — самый простой и быстрый вариант. Можно взять, например, 30-50 компонент вместо 300, при этом сохранив большую часть информации.
        """
        print("df_copy type:", type(df_copy))
        print("df_copy columns:", df_copy.columns if df_copy is not None else None)
        print("'name_vec' in df_copy:", 'name_vec' in df_copy.columns if df_copy is not None else False)
        print("df_copy['name_vec']:", df_copy['name_vec'] if ('name_vec' in df_copy.columns) else "Column not found")
        if df_copy is not None and not df_copy.empty and 'name_vec' in df_copy.columns:
            val = df_copy['name_vec'].iloc[0]
            print("value =", val, "type =", type(val))
        else:
            print("df_copy is None, empty, or missing 'name_vec' column")
        print("value =", val, "type =", type(val))
        print(df_copy['name_vec'].isnull().sum())
        print(df_copy['description_vecs'].isnull().sum())
        print(df_copy['brand_vecs'].isnull().sum())
        print("type(df_copy['name_vec'].iloc[0]) =", type(df_copy['name_vec'].iloc[0]))
        print("df_copy['name_vec'].iloc[0] =", df_copy['name_vec'].iloc[0])

        # Обучаем PCA на эмбеддингах
        pca_name = PCA(n_components=30).fit(df_copy['name_vec'].tolist())
        pca_desc = PCA(n_components=30).fit(df_copy['description_vecs'].tolist())
        pca_brand = PCA(n_components=30).fit(df_copy['brand_vecs'].tolist())

        print("type(pca_name) =", type(pca_name))
        print("pca_name =", pca_name)

        # Сохраняем PCA
        joblib.dump(pca_name, 'pca_name.pkl')
        joblib.dump(pca_desc, 'pca_desc.pkl')
        joblib.dump(pca_brand, 'pca_brand.pkl')

        df_copy['name_vec'] = df_copy['name_vec'].apply(lambda x: list(x))
        df_copy['description_vecs'] = df_copy['description_vecs'].apply(lambda x: list(x))
        df_copy['brand_vecs'] = df_copy['brand_vecs'].apply(lambda x: list(x))

        reduced_embeddings_name = pca_name.transform(df_copy['name_vec'].tolist())
        reduced_embeddings_desc = pca_desc.transform(df_copy['description_vecs'].tolist())
        reduced_embeddings_brand = pca_brand.transform(df_copy['brand_vecs'].tolist())

        # Разворачиваем вектора в отдельные колонки
        embeddings_name_df = pd.DataFrame(reduced_embeddings_name, columns=[f"name_vec_{i+1}" for i in range(30)])
        embeddings_desc_df = pd.DataFrame(reduced_embeddings_desc, columns=[f'description_vecs_{i+1}' for i in range(30)])
        embeddings_brand_df = pd.DataFrame(reduced_embeddings_brand, columns=[f'brand_vecs_{i+1}'for i in range(30)])

        # Объединяем с остальными признаками
        df_copy = pd.concat([df_copy, embeddings_name_df, embeddings_desc_df, embeddings_brand_df], axis=1)

        features = ['rating_1_count', 'rating_2_count', 'rating_3_count', 'rating_4_count', 'rating_5_count', 'comments_published_count', 'photos_published_count',
        'videos_published_count', 'PriceDiscounted', 'item_count_fake_returns7', 'item_count_fake_returns30', 'item_count_fake_returns90', 'item_count_sales7',
        'item_count_sales30', 'item_count_sales90', 'item_count_returns7', 'item_count_returns30', 'item_count_returns90', 'GmvTotal7', 'GmvTotal30', 'GmvTotal90',
        'ExemplarAcceptedCountTotal7', 'ExemplarAcceptedCountTotal30', 'ExemplarAcceptedCountTotal90', 'OrderAcceptedCountTotal7', 'OrderAcceptedCountTotal30',
        'OrderAcceptedCountTotal90', 'ExemplarReturnedCountTotal7', 'ExemplarReturnedCountTotal30', 'ExemplarReturnedCountTotal90', 'ExemplarReturnedValueTotal7',
        'ExemplarReturnedValueTotal30', 'ExemplarReturnedValueTotal90', 'ItemVarietyCount', 'ItemAvailableCount', 'item_time_alive', 'seller_time_alive',
        'item_return_rate_7', 'item_return_rate_30', 'item_return_rate_90', 'item_return_fake_rate_7', 'item_return_fake_rate_30', 'item_return_fake_rate_90',
        'return_fake_vs_real_ratio_7', 'return_fake_vs_real_ratio_30', 'return_fake_vs_real_ratio_90', 'order_return_rate_7', 'order_return_rate_30', 'order_return_rate_90',
        'item_Exemplar_diff_return_7', 'item_Exemplar_diff_return_30', 'item_Exemplar_diff_return_90', 'return_value_7', 'return_value_30', 'return_value_90',
        'return_rate_growth_7_30', 'return_rate_growth_30_90', 'return_rate_7_vs_category_median', 'return_rate_30_vs_category_median', 'return_rate_90_vs_category_median',
        'total_rating', 'avg_rating', 'bad_rating_ratio', '5_star_rating_ratio', 'bad_to_good_rating_ratio', 'comments_per_rating', 'photos_per_rating',
        'videos_per_rating', 'rating_vs_category_median', 'gmv_per_order_7', 'gmv_per_order_30', 'gmv_per_order_90', 'gmv_per_item_7', 'gmv_per_item_30',
        'gmv_per_item_90', 'gmv_per_seller_time_alive_7', 'gmv_per_seller_time_alive_30', 'gmv_per_seller_time_alive_90', 'gmv_growth_7_30', 'gmv_growth_30_90',
        'gmv_per_item_vs_category_median_7', 'gmv_per_item_vs_category_median_30', 'gmv_per_item_vs_category_median_90', 'seller_time_per_item_time', 'seller_items_count',
        'avg_item_time_alive_per_seller', 'avg_orders_per_day_7', 'avg_orders_per_day_30', 'avg_orders_per_day_90', 'avg_price_per_seller', 'price_vs_category_median',
        'seller_total_items', 'seller_items_with_returns', 'seller_items_with_returns_ratio', 'avg_fake_return_ratio_per_seller_7', 'avg_fake_return_ratio_per_seller_30',
        'avg_fake_return_ratio_per_seller_90', 'sales_per_seller_item', 'max_item_time_alive', 'min_item_time_alive', 'sku_variety_per_seller', 'sku_variety_normalized',
        'variety_to_available_ratio', 'avg_available_per_seller', 'zero_available_items_per_seller', 'zero_available_ratio', 'unique_items_per_seller',
        'unavailable_ratio_in_variety', 'price_median_price_category_ratio', 'price_median_price_brand_ratio', 'log_price', 'price_diff_from_category_median',
        'price_diff_from_brand_median', 'price_zscore_category', 'price_zscore_brand',
        'frequency_brand', 'frequency_seller', 'return_rate_brand_7', 'return_rate_brand_30',
        'return_rate_brand_90', 'return_rate_seller_7', 'return_rate_seller_30', 'return_rate_seller_90', 'unique_items_per_brand', 'repeat_items_ratio_brand',
        'repeat_items_ratio_seller', 'brand_similarity_max', 'description_cluster_size', 'description_distance_to_center', 'log_description_distance_to_center',
        'umap_1', 'umap_2', 'description_length', 'name_vec_1', 'name_vec_2', 'name_vec_3', 'name_vec_4', 'name_vec_5', 'name_vec_6',
        'name_vec_7', 'name_vec_8', 'name_vec_9', 'name_vec_10', 'name_vec_11', 'name_vec_12', 'name_vec_13', 'name_vec_14', 'name_vec_15', 'name_vec_16', 'name_vec_17',
        'name_vec_18', 'name_vec_19', 'name_vec_20', 'name_vec_21', 'name_vec_22', 'name_vec_23', 'name_vec_24', 'name_vec_25', 'name_vec_26', 'name_vec_27', 'name_vec_28',
        'name_vec_29', 'name_vec_30', 'description_vecs_1', 'description_vecs_2', 'description_vecs_3', 'description_vecs_4', 'description_vecs_5', 'description_vecs_6',
        'description_vecs_7', 'description_vecs_8', 'description_vecs_9', 'description_vecs_10', 'description_vecs_11', 'description_vecs_12', 'description_vecs_13',
        'description_vecs_14', 'description_vecs_15', 'description_vecs_16', 'description_vecs_17', 'description_vecs_18', 'description_vecs_19', 'description_vecs_20',
        'description_vecs_21', 'description_vecs_22', 'description_vecs_23', 'description_vecs_24', 'description_vecs_25', 'description_vecs_26', 'description_vecs_27',
        'description_vecs_28', 'description_vecs_29', 'description_vecs_30', 'brand_vecs_1', 'brand_vecs_2', 'brand_vecs_3', 'brand_vecs_4', 'brand_vecs_5', 'brand_vecs_6',
        'brand_vecs_7', 'brand_vecs_8', 'brand_vecs_9', 'brand_vecs_10', 'brand_vecs_11', 'brand_vecs_12', 'brand_vecs_13', 'brand_vecs_14', 'brand_vecs_15', 'brand_vecs_16',
        'brand_vecs_17', 'brand_vecs_18', 'brand_vecs_19', 'brand_vecs_20', 'brand_vecs_21', 'brand_vecs_22', 'brand_vecs_23', 'brand_vecs_24', 'brand_vecs_25',
        'brand_vecs_26', 'brand_vecs_27', 'brand_vecs_28', 'brand_vecs_29', 'brand_vecs_30','name_count_ner_description', 'name_count_ner_name_rus',
        'price_count_ner_description', 'price_count_ner_name_rus', 'category_count_ner_description', 'category_count_ner_name_rus', 'brand_count_ner_description', 'brand_count_ner_name_rus',
        'country_count_ner_description', 'country_count_ner_name_rus', 'unique_name_count_ner_description', 'unique_name_count_ner_name_rus', 'unique_price_count_ner_description',
        'unique_price_count_ner_name_rus', 'avg_conf_name_ner_description', 'avg_conf_name_ner_name_rus', 'max_conf_price_ner_description', 'max_conf_price_ner_name_rus',
        'var_conf_category_ner_description', 'var_conf_category_ner_name_rus', 'unique_tag_count_description', 'unique_tag_count_name_rus', 'price_extracted',
        'price_diff_from_brand_median_ner']

        categ_features = ['brand_name', 'SellerID', 'CommercialCategory', 'returns_without_sales_7', 'returns_without_sales_30', 'returns_without_sales_90',
        'no_rating', 'few_rating', 'gmv_without_comments_7', 'gmv_without_comments_30', 'gmv_without_comments_90', 'gmv_without_rating_7', 'gmv_without_rating_30',
        'gmv_without_rating_90', 'gmv_without_photos_7', 'gmv_without_photos_30', 'gmv_without_photos_90', 'low_variety', 'gmv_with_low_rating_7', 'gmv_with_low_rating_30',
        'gmv_with_low_rating_90', 'young_seller', 'seller_age_bucket', 'low_variety_count', 'high_variety_count', 'sales_without_availability',
        'is_cheaper_than_median_category', 'is_cheaper_than_median_brand', 'name_cluster', 'is_outlier_name', 'description_cluster', 'is_outlier_description',
        'has_name_ner_description', 'has_name_ner_name_rus', 'has_price_ner_description', 'has_price_ner_name_rus', 'has_category_ner_description', 'has_category_ner_name_rus',
        'has_brand_ner_description', 'has_brand_ner_name_rus', 'has_country_ner_description', 'has_country_ner_name_rus', 'name_match_desc_name', 'price_match_desc_name',
        'category_match_desc_name', 'brand_match_desc_name', 'country_match_desc_name', 'brand_in_desc_not_in_name', 'category_in_name_not_in_desc',
        'has_any_entity_description', 'has_any_entity_name_rus', 'multiple_brands_in_desc']

        for col in categ_features:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].astype(str).fillna('nan')
            else:
                print(f"[INFO] Колонка '{col}' отсутствует. Пропускаем.")

        full_features = features + categ_features
        existing_cols = [col for col in ['resolution'] + full_features if col in df_copy.columns]
        df_copy = df_copy.dropna(subset=existing_cols)

        X = df_copy[full_features]
        y = df_copy['resolution']

        categ_features_indices = [X.columns.get_loc(cat) for cat in categ_features]

        print('Nans in y(resolution):', y.isna().sum())
        print('Nans in X:')
        print(X.isna().sum().sort_values(ascending=False).head(10))

        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = CatBoostClassifier(random_state=42)
        model.fit(X_train, Y_train, cat_features=categ_features_indices, eval_set=(X_test, Y_test), verbose=100)

        # Сохраняем модель
        model.save_model("CatBoost_model.cbm")

        # Оцениваем
        Y_pred = model.predict(X_test)
        Y_proba = model.predict_proba(X_test)[:,1]
        print(classification_report(Y_test, Y_pred))
        if len(set(Y_test))>1:
            print("AUC: ", roc_auc_score(Y_test, Y_proba))
        else:
            print('Только один класс в Y_test, AUC не определен')

        df_copy = df_copy.copy()
        return df_copy

    def prepare_features(self, df_copy):
        """
        Трансформация новых данных без переобучения
        """
        # Проверяем, нужно ли применять PCA
        need_pca = all(col in df_copy.columns for col in ['name_vec', 'description_vecs', 'brand_vecs'])

        if need_pca:
            print("Применяем PCA к эмбеддингам...")
            # Загружаем обученные PCA
            pca_name = load('pca_name.pkl')
            pca_desc = load('pca_desc.pkl')
            pca_brand = load('pca_brand.pkl')

            # Преобразуем эмбеддинги
            reduced_embeddings_name = pca_name.transform(df_copy['name_vec'].tolist())
            reduced_embeddings_desc = pca_desc.transform(df_copy['description_vecs'].tolist())
            reduced_embeddings_brand = pca_brand.transform(df_copy['brand_vecs'].tolist())

            # Разворачиваем вектора в отдельные колонки
            embeddings_name_df = pd.DataFrame(reduced_embeddings_name, columns=[f"name_vec_{i + 1}" for i in range(30)])
            embeddings_desc_df = pd.DataFrame(reduced_embeddings_desc, columns=[f'description_vecs_{i + 1}' for i in range(30)])
            embeddings_brand_df = pd.DataFrame(reduced_embeddings_brand, columns=[f'brand_vecs_{i + 1}' for i in range(30)])

            # Объединяем с остальными признаками
            df_copy = df_copy.drop(columns=['name_vec', 'description_vecs', 'brand_vecs'])
            df_copy = pd.concat([df_copy, embeddings_name_df, embeddings_desc_df, embeddings_brand_df], axis=1)
        else:
            print("Эмбеддинги отсутствуют. Предполагаем, что PCA-признаки уже есть.")

        df_copy = df_copy.copy()
        return df_copy

    def predict(self, df):
        """
        Загрузка модели и предсказание
        """
        # Готовим фичи
        df_prepared = self.prepare_features(df)

        features = ['rating_1_count', 'rating_2_count', 'rating_3_count', 'rating_4_count', 'rating_5_count',
                    'comments_published_count', 'photos_published_count',
                    'videos_published_count', 'PriceDiscounted', 'item_count_fake_returns7',
                    'item_count_fake_returns30', 'item_count_fake_returns90', 'item_count_sales7',
                    'item_count_sales30', 'item_count_sales90', 'item_count_returns7', 'item_count_returns30',
                    'item_count_returns90', 'GmvTotal7', 'GmvTotal30', 'GmvTotal90',
                    'ExemplarAcceptedCountTotal7', 'ExemplarAcceptedCountTotal30', 'ExemplarAcceptedCountTotal90',
                    'OrderAcceptedCountTotal7', 'OrderAcceptedCountTotal30',
                    'OrderAcceptedCountTotal90', 'ExemplarReturnedCountTotal7', 'ExemplarReturnedCountTotal30',
                    'ExemplarReturnedCountTotal90', 'ExemplarReturnedValueTotal7',
                    'ExemplarReturnedValueTotal30', 'ExemplarReturnedValueTotal90', 'ItemVarietyCount',
                    'ItemAvailableCount', 'item_time_alive', 'seller_time_alive',
                    'item_return_rate_7', 'item_return_rate_30', 'item_return_rate_90', 'item_return_fake_rate_7',
                    'item_return_fake_rate_30', 'item_return_fake_rate_90',
                    'return_fake_vs_real_ratio_7', 'return_fake_vs_real_ratio_30', 'return_fake_vs_real_ratio_90',
                    'order_return_rate_7', 'order_return_rate_30', 'order_return_rate_90',
                    'item_Exemplar_diff_return_7', 'item_Exemplar_diff_return_30', 'item_Exemplar_diff_return_90',
                    'return_value_7', 'return_value_30', 'return_value_90',
                    'return_rate_growth_7_30', 'return_rate_growth_30_90', 'return_rate_7_vs_category_median',
                    'return_rate_30_vs_category_median', 'return_rate_90_vs_category_median',
                    'total_rating', 'avg_rating', 'bad_rating_ratio', '5_star_rating_ratio', 'bad_to_good_rating_ratio',
                    'comments_per_rating', 'photos_per_rating',
                    'videos_per_rating', 'rating_vs_category_median', 'gmv_per_order_7', 'gmv_per_order_30',
                    'gmv_per_order_90', 'gmv_per_item_7', 'gmv_per_item_30',
                    'gmv_per_item_90', 'gmv_per_seller_time_alive_7', 'gmv_per_seller_time_alive_30',
                    'gmv_per_seller_time_alive_90', 'gmv_growth_7_30', 'gmv_growth_30_90',
                    'gmv_per_item_vs_category_median_7', 'gmv_per_item_vs_category_median_30',
                    'gmv_per_item_vs_category_median_90', 'seller_time_per_item_time', 'seller_items_count',
                    'avg_item_time_alive_per_seller', 'avg_orders_per_day_7', 'avg_orders_per_day_30',
                    'avg_orders_per_day_90', 'avg_price_per_seller', 'price_vs_category_median',
                    'seller_total_items', 'seller_items_with_returns', 'seller_items_with_returns_ratio',
                    'avg_fake_return_ratio_per_seller_7', 'avg_fake_return_ratio_per_seller_30',
                    'avg_fake_return_ratio_per_seller_90', 'sales_per_seller_item', 'max_item_time_alive',
                    'min_item_time_alive', 'sku_variety_per_seller', 'sku_variety_normalized',
                    'variety_to_available_ratio', 'avg_available_per_seller', 'zero_available_items_per_seller',
                    'zero_available_ratio', 'unique_items_per_seller',
                    'unavailable_ratio_in_variety', 'price_median_price_category_ratio',
                    'price_median_price_brand_ratio', 'log_price', 'price_diff_from_category_median',
                    'price_diff_from_brand_median', 'price_zscore_category', 'price_zscore_brand',
                    'frequency_brand', 'frequency_seller',
                    'return_rate_brand_7', 'return_rate_brand_30',
                    'return_rate_brand_90', 'return_rate_seller_7', 'return_rate_seller_30', 'return_rate_seller_90',
                    'unique_items_per_brand', 'repeat_items_ratio_brand',
                    'repeat_items_ratio_seller', 'brand_similarity_max', 'description_cluster_size',
                    'description_distance_to_center', 'log_description_distance_to_center',
                    'umap_1', 'umap_2', 'description_length',  'name_vec_1', 'name_vec_2',
                    'name_vec_3', 'name_vec_4', 'name_vec_5', 'name_vec_6',
                    'name_vec_7', 'name_vec_8', 'name_vec_9', 'name_vec_10', 'name_vec_11', 'name_vec_12',
                    'name_vec_13', 'name_vec_14', 'name_vec_15', 'name_vec_16', 'name_vec_17',
                    'name_vec_18', 'name_vec_19', 'name_vec_20', 'name_vec_21', 'name_vec_22', 'name_vec_23',
                    'name_vec_24', 'name_vec_25', 'name_vec_26', 'name_vec_27', 'name_vec_28',
                    'name_vec_29', 'name_vec_30', 'description_vecs_1', 'description_vecs_2', 'description_vecs_3',
                    'description_vecs_4', 'description_vecs_5', 'description_vecs_6',
                    'description_vecs_7', 'description_vecs_8', 'description_vecs_9', 'description_vecs_10',
                    'description_vecs_11', 'description_vecs_12', 'description_vecs_13',
                    'description_vecs_14', 'description_vecs_15', 'description_vecs_16', 'description_vecs_17',
                    'description_vecs_18', 'description_vecs_19', 'description_vecs_20',
                    'description_vecs_21', 'description_vecs_22', 'description_vecs_23', 'description_vecs_24',
                    'description_vecs_25', 'description_vecs_26', 'description_vecs_27',
                    'description_vecs_28', 'description_vecs_29', 'description_vecs_30', 'brand_vecs_1', 'brand_vecs_2',
                    'brand_vecs_3', 'brand_vecs_4', 'brand_vecs_5', 'brand_vecs_6',
                    'brand_vecs_7', 'brand_vecs_8', 'brand_vecs_9', 'brand_vecs_10', 'brand_vecs_11', 'brand_vecs_12',
                    'brand_vecs_13', 'brand_vecs_14', 'brand_vecs_15', 'brand_vecs_16',
                    'brand_vecs_17', 'brand_vecs_18', 'brand_vecs_19', 'brand_vecs_20', 'brand_vecs_21',
                    'brand_vecs_22', 'brand_vecs_23', 'brand_vecs_24', 'brand_vecs_25',
                    'brand_vecs_26', 'brand_vecs_27', 'brand_vecs_28', 'brand_vecs_29', 'brand_vecs_30', 'name_count_ner_description', 'name_count_ner_name_rus',
                    'price_count_ner_description', 'price_count_ner_name_rus', 'category_count_ner_description', 'category_count_ner_name_rus', 'brand_count_ner_description', 'brand_count_ner_name_rus',
                    'country_count_ner_description', 'country_count_ner_name_rus', 'unique_name_count_ner_description', 'unique_name_count_ner_name_rus', 'unique_price_count_ner_description',
                    'unique_price_count_ner_name_rus', 'avg_conf_name_ner_description', 'avg_conf_name_ner_name_rus', 'max_conf_price_ner_description', 'max_conf_price_ner_name_rus',
                    'var_conf_category_ner_description', 'var_conf_category_ner_name_rus', 'unique_tag_count_description', 'unique_tag_count_name_rus', 'price_extracted',
                    'price_diff_from_brand_median_ner']

        missing_features = [
            'fake_items_from_seller_count', 'fake_items_from_seller_ratio',
            'fake_items_from_brand_count', 'fake_items_from_brand_ratio',
            'logits_description'
        ]

        categ_features = ['brand_name', 'SellerID', 'CommercialCategory', 'returns_without_sales_7',
                          'returns_without_sales_30', 'returns_without_sales_90',
                          'no_rating', 'few_rating', 'gmv_without_comments_7', 'gmv_without_comments_30',
                          'gmv_without_comments_90', 'gmv_without_rating_7', 'gmv_without_rating_30',
                          'gmv_without_rating_90', 'gmv_without_photos_7', 'gmv_without_photos_30',
                          'gmv_without_photos_90', 'low_variety', 'gmv_with_low_rating_7', 'gmv_with_low_rating_30',
                          'gmv_with_low_rating_90', 'young_seller', 'seller_age_bucket', 'low_variety_count',
                          'high_variety_count', 'sales_without_availability',
                          'is_cheaper_than_median_category', 'is_cheaper_than_median_brand', 'name_cluster',
                          'is_outlier_name', 'description_cluster', 'is_outlier_description',
                            'has_name_ner_description', 'has_name_ner_name_rus', 'has_price_ner_description', 'has_price_ner_name_rus', 'has_category_ner_description', 'has_category_ner_name_rus',
                            'has_brand_ner_description', 'has_brand_ner_name_rus', 'has_country_ner_description', 'has_country_ner_name_rus', 'name_match_desc_name', 'price_match_desc_name',
                            'category_match_desc_name', 'brand_match_desc_name', 'country_match_desc_name', 'brand_in_desc_not_in_name', 'category_in_name_not_in_desc',
                            'has_any_entity_description', 'has_any_entity_name_rus', 'multiple_brands_in_desc']

        features = [f for f in features if f not in missing_features]
        all_features = features + categ_features
        all_features = list(dict.fromkeys(all_features))  # Удаляем дубликаты
        X = df_prepared[all_features]

        X = X.copy()
        for col in categ_features:
            if col in X.columns:
                X[col] = X[col].astype(str).fillna('missing')
            else:
                print(f"Колонка {col} отсутствует в X.columns!")

        if X.columns.duplicated().any():
            print("В X.columns есть дубликаты! Удаляем их.")
            X = X.loc[:, ~X.columns.duplicated()]

        model = CatBoostClassifier()

        # Загружаем модель
        if not os.path.exists("CatBoost_model.cbm"):
            raise FileNotFoundError("Модель не найдена: CatBoost_model.cbm")
        else:
            model.load_model("CatBoost_model.cbm", format='cbm')

        #categ_features_indices = [X.columns.get_loc(cat) for cat in categ_features]
        categ_features_indices = [list(X.columns).index(cat) for cat in categ_features if cat in X.columns]

        pool = Pool(data=X, cat_features=categ_features_indices)
        # Оцениваем
        Y_pred = model.predict(pool)
        Y_proba = model.predict_proba(pool)[:, 1]

        return Y_pred, Y_proba


    """
    1. Анализ возвратов
    Контрафакт часто сопровождается большим числом возвратов.
    
    Полезные фичи:
    
    item_count_returns7/30/90 vs item_count_sales7/30/90 → return_rate_X = returns / sales
    item_count_fake_returnsX → может быть прокси на скручивания возвратов
    ExemplarReturnedCountTotalX / ExemplarAcceptedCountTotalX → возвраты на уровне заказов
    
    Что смотреть:
    
    Высокий return_rate
    Необычно много фиктивных возвратов
    Разница между возвратами по item_* и ExemplarReturned* → потенциальная нечестность в отчетности
    """
    def feature_engineering_returns(self, df_copy):
        """
        У нормального товара — возврат обычно в пределах 3–10%
        У контрафакта — может быть 30%, 50% и выше
        """
        df_copy.loc[:, 'item_return_rate_7'] = df_copy['item_count_returns7'] / (df_copy['item_count_sales7']+1)
        df_copy.loc[:, 'item_return_rate_30'] = df_copy['item_count_returns30'] / (df_copy['item_count_sales30'] + 1)
        df_copy.loc[:, 'item_return_rate_90'] = df_copy['item_count_returns90'] / (df_copy['item_count_sales90'] + 1)
        """
        Это могут быть попытки "замести следы": удалить плохие отзывы, откатить некачественную продажу
        Или возвраты, созданные самими продавцами — в схемах по скрутке рейтинга или сбыту подделок
        """
        df_copy.loc[:, 'item_return_fake_rate_7'] = df_copy['item_count_fake_returns7'] / (df_copy['item_count_sales7'] + 1)
        df_copy.loc[:, 'item_return_fake_rate_30'] = df_copy['item_count_fake_returns30'] / (df_copy['item_count_sales30'] + 1)
        df_copy.loc[:, 'item_return_fake_rate_90'] = df_copy['item_count_fake_returns90'] / (df_copy['item_count_sales90'] + 1)
        """
        Большая доля фиктивных возвратов по сравнению с общими
        """
        df_copy.loc[:, 'return_fake_vs_real_ratio_7'] = df_copy['item_return_fake_rate_7'] / (df_copy['item_return_rate_7'] +1)
        df_copy.loc[:, 'return_fake_vs_real_ratio_30'] = df_copy['item_return_fake_rate_30'] / (df_copy['item_return_rate_30'] + 1)
        df_copy.loc[:, 'return_fake_vs_real_ratio_90'] = df_copy['item_return_fake_rate_90'] / (df_copy['item_return_rate_90'] + 1)
        """
        Возвраты в рамках заказов (Exemplar = товарная единица):
        ExemplarAcceptedCountTotalX — сколько всего товаров приняли в заказе
        ExemplarReturnedCountTotalX — сколько из них вернули
        Почему это важно:
        Это "глобальный" уровень — агрегировано по заказам, а не по товарным SKU
        Можно увидеть, насколько общая доля возвратов среди проданных товаров велика
        """
        df_copy.loc[:, 'order_return_rate_7'] = df_copy['ExemplarReturnedCountTotal7'] / (df_copy['ExemplarAcceptedCountTotal7']+1)
        df_copy.loc[:, 'order_return_rate_30'] = df_copy['ExemplarReturnedCountTotal30'] / (df_copy['ExemplarAcceptedCountTotal30'] + 1)
        df_copy.loc[:, 'order_return_rate_90'] = df_copy['ExemplarReturnedCountTotal90'] / (df_copy['ExemplarAcceptedCountTotal90'] + 1)
        """
        Если один источник (item_count_returnsX) показывает одно, а другой (ExemplarReturnedCountTotalX) — другое, 
        это может быть индикатор махинаций, особенно у продавца с плохой репутацией.
        Положительные значения → больше возвратов на уровне товара, чем заказов
        Отрицательные → наоборот — странно, может говорить о несоответствии учёта
        """
        df_copy.loc[:, 'item_Exemplar_diff_return_7'] = (df_copy['item_count_returns7'] - df_copy['ExemplarReturnedCountTotal7']) / (df_copy['item_count_sales7']+1)
        df_copy.loc[:, 'item_Exemplar_diff_return_30'] = (df_copy['item_count_returns30'] - df_copy['ExemplarReturnedCountTotal30']) / (df_copy['item_count_sales30']+1)
        df_copy.loc[:, 'item_Exemplar_diff_return_90'] = (df_copy['item_count_returns90'] - df_copy['ExemplarReturnedCountTotal90']) / (df_copy['item_count_sales90']+1)
        """
        Отношение возвратов к GMV (Gross Merchandise Value)
        Контрафакт может быть дешевым товаром с высоким объёмом возвратов, и доля возвратов по стоимости даст другой взгляд на проблему.
        """
        df_copy.loc[:, 'return_value_7'] = df_copy['ExemplarReturnedValueTotal7'] / (df_copy['GmvTotal7']+1)
        df_copy.loc[:, 'return_value_30'] = df_copy['ExemplarReturnedValueTotal30'] / (df_copy['GmvTotal30'] + 1)
        df_copy.loc[:, 'return_value_90'] = df_copy['ExemplarReturnedValueTotal90'] / (df_copy['GmvTotal90'] + 1)
        """
        Временные изменения: резкие скачки возвратов
        Контрафакт может не сразу выявляться. Если возвраты резко выросли за последнюю неделю — тревожный сигнал.
        """
        df_copy.loc[:, 'return_rate_growth_7_30'] = (df_copy['item_return_rate_7'] - df_copy['item_return_rate_30']) / (df_copy['item_return_rate_30'] +1)
        df_copy.loc[:, 'return_rate_growth_30_90'] = (df_copy['item_return_rate_30'] - df_copy['item_return_rate_90']) / (df_copy['item_return_rate_90'] + 1)
        """
        Сравнение с медианой/средним по категории
        Если товар сильно выбивается из средней по категории, это может быть индикатор.
        Значения >>1 → возврат товара выше нормы по категории
        """
        df_copy.loc[:, 'return_rate_7_vs_category_median'] = df_copy['item_return_rate_7']/(df_copy.groupby('CommercialCategory')['item_return_rate_7'].transform('median')+1)
        df_copy.loc[:, 'return_rate_30_vs_category_median'] = df_copy['item_return_rate_30'] / (df_copy.groupby('CommercialCategory')['item_return_rate_30'].transform('median') + 1)
        df_copy.loc[:, 'return_rate_90_vs_category_median'] = df_copy['item_return_rate_90'] / (df_copy.groupby('CommercialCategory')['item_return_rate_90'].transform('median') + 1)
        """
        Признаки отсутствия продаж, но наличие возвратов
        Это может быть ошибкой или злоупотреблением.
        """
        df_copy.loc[:, 'returns_without_sales_7'] = ((df_copy['item_count_returns7'] > 0 ) & ( df_copy['item_count_sales7'] == 0)).astype(int)
        df_copy.loc[:, 'returns_without_sales_30'] = ((df_copy['item_count_returns30'] > 0) & (df_copy['item_count_sales30'] == 0)).astype(int)
        df_copy.loc[:, 'returns_without_sales_90'] = ((df_copy['item_count_returns90'] > 0) & (df_copy['item_count_sales90'] == 0)).astype(int)
        """
        item_return_rate_X: item_count_returnsX / item_count_salesX
        — базовый показатель уровня возвратов
        
        item_return_fake_rate_X: item_count_fake_returnsX / item_count_salesX
        — доля подозрительных возвратов
        
        order_return_rate_X: ExemplarReturnedCountTotalX / ExemplarAcceptedCountTotalX
        — возвраты на уровне заказов
        
        return_value_X: ExemplarReturnedValueTotalX / GmvTotalX
        — возвраты по деньгам
        
        return_rate_growth_X_Y: return_rate_X - return_rate_Y
        — всплески возвратов
        
        return_rate_X_vs_category_median: return_rate_X / median(category)
        — сравнение с категорией
        
        returns_without_sales_X: есть возвраты, но нет продаж
        — подозрительное поведение
        
        item_Exemplar_diff_return_X: item vs order return discrepancy
        — различие между item-level и order-level возвратами
        
         Что искать:
        
        Высокий return_rate
        Аномально много фиктивных возвратов
        Высокая доля возвратов по стоимости (GMV)
        Внезапный рост возвратов за 7 дней    
        Несоответствие между возвратами на уровне товара и заказов   
        Возвраты при отсутствии продаж
        """
        df_copy = df_copy.copy()
        return df_copy
    """
    2. Поведение рейтингов
    Если присутствует, то может быть индикатором качества.
    
    Фичи:
    
    Средний рейтинг, например:
    avg_rating = (1*r1 + 2*r2 + ... + 5*r5) / (r1 + r2 + ... + r5)
    
    Процент негативных рейтингов:
    bad_rating_ratio = (r1 + r2) / total_ratings
    """
    def feature_engineering_ratings(self, df_copy):
        df_copy.loc[:, 'total_rating'] = df_copy['rating_1_count']+df_copy['rating_2_count']+df_copy['rating_3_count']+df_copy['rating_4_count']+df_copy['rating_5_count']
        df_copy.loc[:, 'avg_rating'] = (1 * df_copy['rating_1_count'] + 2*df_copy['rating_2_count'] +3*df_copy['rating_3_count'] +4*df_copy['rating_4_count']+5*df_copy['rating_5_count'])/(df_copy['total_rating']+1)
        df_copy.loc[:, 'bad_rating_ratio'] = (df_copy['rating_1_count'] +df_copy['rating_2_count'])/(df_copy['total_rating']+1)
        """
        Доля 5-звездочных оценок
        Контрафакт может иметь "накрученные" 5-звездочные оценки, особенно если это единственные положительные.
        """
        df_copy.loc[:, '5_star_rating_ratio'] = df_copy['rating_5_count'] / (df_copy['total_rating']+1)
        """
        Наличие слишком малого количества отзывов (или ноль)
        Это может сигнализировать, что продавец старается скрыть реальные отзывы.
        """
        df_copy.loc[:, 'no_rating'] = (df_copy['total_rating'] == 0 ).astype(int)
        df_copy.loc[:, 'few_rating'] = (df_copy['total_rating'] < 3 ).astype(int)
        """
        Соотношение плохих к хорошим
        Если плохих больше, чем хороших — это тревожный звоночек.
        """
        df_copy.loc[:, 'bad_to_good_rating_ratio'] = (df_copy['rating_1_count'] + df_copy['rating_2_count'])/(df_copy['rating_5_count']+df_copy['rating_4_count']+1)
        """
        Комментарии и медиаактивность
        Если у товара много плохих оценок, но мало комментариев и фото, это подозрительно
        """
        df_copy.loc[:, 'comments_per_rating'] = df_copy['comments_published_count']/(df_copy['total_rating']+1)
        df_copy.loc[:, 'photos_per_rating'] = df_copy['photos_published_count'] / (df_copy['total_rating'] + 1)
        df_copy.loc[:, 'videos_per_rating'] = df_copy['videos_published_count'] / (df_copy['total_rating'] + 1)
        """
        Сравнение с категорией
        Как и в случае возвратов, полезно сравнить средний рейтинг с медианным по CommercialTypeName4
        """
        df_copy.loc[:, 'rating_vs_category_median'] = df_copy['avg_rating']/(df_copy.groupby('CommercialCategory')['avg_rating'].transform('median') +1)

        df_copy = df_copy.copy()
        return df_copy

    """
    3. Продажи и GMV
    Анализируем краткосрочные пики и подозрительную активность.
    
    Признаки:
    Слишком большой GMV на фоне малоактивного продавца
    Высокие продажи, но резкий всплеск возвратов
    GMV без комментариев или оценок → возможно накрутка
    """
    def feature_engineering_sales(self, df_copy):
        """
        GMV на 1 заказ / 1 товар — может показать завышенную цену
        Сколько в среднем зарабатывает продавец или товар с одного принятого заказа за последние 7 дней.
        Если GMV очень высокий, но заказов мало — это может говорить о:
        искусственно завышенной цене,
        неестественной накрутке GMV,
        или вбросе больших сумм за один-два заказа (типичная схема для накруток).  
        Если значение резко отличается от медианы по категории — тоже может быть тревожным индикатором.
        """
        df_copy.loc[:, 'gmv_per_order_7'] = df_copy['GmvTotal7'] / (df_copy['OrderAcceptedCountTotal7']+1)
        df_copy.loc[:, 'gmv_per_order_30'] = df_copy['GmvTotal30'] / (df_copy['OrderAcceptedCountTotal30'] + 1)
        df_copy.loc[:, 'gmv_per_order_90'] = df_copy['GmvTotal90'] / (df_copy['OrderAcceptedCountTotal90'] + 1)

        df_copy.loc[:, 'gmv_per_item_7'] = df_copy['GmvTotal7'] / (df_copy['ExemplarAcceptedCountTotal7']+1)
        df_copy.loc[:, 'gmv_per_item_30'] = df_copy['GmvTotal30'] / (df_copy['ExemplarAcceptedCountTotal30'] + 1)
        df_copy.loc[:, 'gmv_per_item_90'] = df_copy['GmvTotal90'] / (df_copy['ExemplarAcceptedCountTotal90'] + 1)
        """
        GMV на фоне активности продавца
        Контрафакт часто продают "вбросами" — создают нового продавца, быстро делают много продаж (GMV) и исчезают.
    
        Если:
        Продавец зарегистрирован недавно
        Но уже сгенерировал большой GMV за короткое время
        → Это подозрительно.
        """
        df_copy.loc[:, 'gmv_per_seller_time_alive_7'] = df_copy['GmvTotal7']/ (df_copy['seller_time_alive']+1)
        df_copy.loc[:, 'gmv_per_seller_time_alive_30'] = df_copy['GmvTotal30'] / (df_copy['seller_time_alive'] + 1)
        df_copy.loc[:, 'gmv_per_seller_time_alive_90'] = df_copy['GmvTotal90'] / (df_copy['seller_time_alive'] + 1)
        """
        GMV при отсутствии рейтингов → подозрительная активность
        """
        df_copy.loc[:, 'gmv_without_rating_7'] = ((df_copy['GmvTotal7'] >0) & (df_copy['total_rating'] == 0)).astype(int)
        df_copy.loc[:, 'gmv_without_rating_30'] = ((df_copy['GmvTotal30'] > 0) & (df_copy['total_rating'] == 0)).astype(int)
        df_copy.loc[:, 'gmv_without_rating_90'] = ((df_copy['GmvTotal90'] > 0) & (df_copy['total_rating'] == 0)).astype(int)

        df_copy.loc[:, 'gmv_without_comments_7'] = ((df_copy['GmvTotal7'] > 0) & (df_copy['comments_published_count'] == 0)).astype(int)
        df_copy.loc[:, 'gmv_without_comments_30'] = ((df_copy['GmvTotal30'] > 0) & (df_copy['comments_published_count'] == 0)).astype(int)
        df_copy.loc[:, 'gmv_without_comments_90'] = ((df_copy['GmvTotal90'] > 0) & (df_copy['comments_published_count'] == 0)).astype(int)

        df_copy.loc[:, 'gmv_without_photos_7'] = ((df_copy['GmvTotal7'] > 0) & (df_copy['photos_published_count'] == 0)).astype(int)
        df_copy.loc[:, 'gmv_without_photos_30'] = ((df_copy['GmvTotal30'] > 0) & (df_copy['photos_published_count'] == 0)).astype(int)
        df_copy.loc[:, 'gmv_without_photos_90'] = ((df_copy['GmvTotal90'] > 0) & (df_copy['photos_published_count'] == 0)).astype(int)
        """
        Рост GMV — всплески
        Как и для возвратов, внезапный рост GMV — возможный индикатор схем
        """
        df_copy.loc[:, 'gmv_growth_7_30'] = (df_copy['GmvTotal7'] - df_copy['GmvTotal30']) / (df_copy['GmvTotal30'] + 1)
        df_copy.loc[:, 'gmv_growth_30_90'] = (df_copy['GmvTotal30'] - df_copy['GmvTotal90']) / (df_copy['GmvTotal90'] + 1)
        """
        GMV против среднего по категории
        Контрафакт часто маскируется под популярные бренды, но завышает цену.
        Сравнение по категории поможет выявить вбросы дорогих подделок.
        """
        df_copy.loc[:, 'gmv_per_item_vs_category_median_7'] = df_copy['gmv_per_item_7'] / (df_copy.groupby('CommercialCategory')['gmv_per_item_7'].transform('median') + 1)
        df_copy.loc[:, 'gmv_per_item_vs_category_median_30'] = df_copy['gmv_per_item_30'] / (df_copy.groupby('CommercialCategory')['gmv_per_item_30'].transform('median') + 1)
        df_copy.loc[:, 'gmv_per_item_vs_category_median_90'] = df_copy['gmv_per_item_90'] / (df_copy.groupby('CommercialCategory')['gmv_per_item_90'].transform('median') + 1)
        """
        Количество разновидностей товара
        Контрафакт может продаваться в одной разновидности (цвет, размер), в отличие от оригинала
        """
        df_copy.loc[:, 'low_variety'] = (df_copy['ItemVarietyCount']<2).astype(int)
        """
        Проверка GMV vs рейтинг
        Можно добавить простую метрику: если GMV большой, а средний рейтинг плохой, это может говорить о подделке с раскруткой
        """
        df_copy.loc[:, 'gmv_with_low_rating_7'] = ((df_copy['GmvTotal7']>0)&(df_copy['avg_rating']<2.5)).astype(int)
        df_copy.loc[:, 'gmv_with_low_rating_30'] = ((df_copy['GmvTotal30'] > 0) & (df_copy['avg_rating'] < 2.5)).astype(int)
        df_copy.loc[:, 'gmv_with_low_rating_90'] = ((df_copy['GmvTotal90'] > 0) & (df_copy['avg_rating'] < 2.5)).astype(int)

        df_copy = df_copy.copy()
        return df_copy

    """
    4. Профиль продавца
    Некоторые признаки "молодых" или подозрительных продавцов.
    
    Фичи:
    seller_time_alive (в днях, часах?) — короткий срок может быть индикатором "однодневок"
    Соотношение seller_time_alive к item_time_alive — если продавец активен меньше времени, чем товар
    """
    def feature_engineering_seller(self, df_copy):
        df_copy.loc[:, 'young_seller'] = (df_copy['seller_time_alive']<50).astype(int)
        df_copy.loc[:, 'seller_age_bucket'] = pd.cut(df_copy['seller_time_alive'], bins=[0, 30, 180, 365, 10000], labels=[1, 2, 3, 4])
        df_copy.loc[:, 'seller_time_per_item_time'] = df_copy['seller_time_alive'] / (df_copy['item_time_alive']+1)
        """
        seller_items_count — количество товаров у продавца
        Это даст каждому товару колонку с числом разных товаров у продавца
        """
        df_copy.loc[:, 'seller_items_count'] = df_copy.groupby('SellerID')['id'].transform('nunique')
        """
        avg_item_time_alive_per_seller — среднее время активности товаров продавца
        Если item_time_alive — количество дней с момента публикации, это покажет, насколько "долго живёт" ассортимент продавца.
        """
        df_copy.loc[:, 'avg_item_time_alive_per_seller'] = df_copy.groupby('SellerID')['item_time_alive'].transform('mean')
        """
        avg_orders_per_day — активность продавца с момента регистрации
        """
        df_copy.loc[:, 'avg_orders_per_day_7'] = df_copy['OrderAcceptedCountTotal7'] / (df_copy['seller_time_alive'] +1)
        df_copy.loc[:, 'avg_orders_per_day_30'] = df_copy['OrderAcceptedCountTotal30'] / (df_copy['seller_time_alive'] + 1)
        df_copy.loc[:, 'avg_orders_per_day_90'] = df_copy['OrderAcceptedCountTotal90'] / (df_copy['seller_time_alive'] + 1)
        """
        Средняя цена товаров продавца (по скидке)
        Если продавец торгует исключительно дешевыми или наоборот — завышенными по цене товарами, это может быть сигналом.
        """
        df_copy.loc[:, 'avg_price_per_seller'] = df_copy.groupby('SellerID')['PriceDiscounted'].transform('mean')
        """
        Можно также сравнить это значение с медианой по категории
        """
        df_copy.loc[:, 'price_vs_category_median'] = df_copy['PriceDiscounted'] / (df_copy.groupby('CommercialCategory')['PriceDiscounted'].transform('median')+1)
        """
        Процент товаров с возвратами у продавца
        Контрафакт часто имеет высокий процент возвратов.
        """
        df_copy.loc[:, 'seller_total_items'] = df_copy.groupby('SellerID')['id'].transform('count')
        df_copy.loc[:, 'seller_items_with_returns'] = df_copy.groupby('SellerID')['item_count_returns30'].transform(lambda x: (x > 0).sum())
        df_copy.loc[:, 'seller_items_with_returns_ratio'] = df_copy['seller_items_with_returns'] / (df_copy['seller_total_items']+1)
        """
        Доля фиктивных возвратов по отношению к продажам
        Если у продавца высокий процент "fake returns", это красный флаг
        Также можно усреднить по продавцу
        """
        df_copy.loc[:, 'avg_fake_return_ratio_per_seller_7'] = df_copy.groupby('SellerID')['item_return_fake_rate_7'].transform('mean')
        df_copy.loc[:, 'avg_fake_return_ratio_per_seller_30'] = df_copy.groupby('SellerID')['item_return_fake_rate_30'].transform('mean')
        df_copy.loc[:, 'avg_fake_return_ratio_per_seller_90'] = df_copy.groupby('SellerID')['item_return_fake_rate_90'].transform('mean')
        """
        Продажи на одного товарa
        Если у продавца один товар, но тысячи продаж — это подозрительно
        """
        df_copy.loc[:, 'sales_per_seller_item'] = df_copy['item_count_sales30'] / (df_copy['seller_items_count']+1)
        """
        Возраст самых "старых" и "новых" товаров у продавца
        Например, у подозрительных продавцов все товары добавлены недавно
        """
        df_copy.loc[:, 'max_item_time_alive'] = df_copy.groupby('SellerID')['item_time_alive'].transform('max')
        df_copy.loc[:, 'min_item_time_alive'] = df_copy.groupby('SellerID')['item_time_alive'].transform('min')

        df_copy = df_copy.copy()
        return df_copy

    """
    5. Доступность и разнообразие
    Контрафактные товары иногда представлены в единичных вариантах или наоборот — "заливаются" в разных вариациях массово.
    
    Фичи:
    ItemVarietyCount — много вариаций одного товара → может быть подделка под оригинал
    ItemAvailableCount — недоступный товар с продажами/возвратами? — подозрительно
    """
    def feature_engineering_availability_variety(self, df_copy):
        """
        Низкое разнообразие может означать "серийную подделку" — один вариант с массовыми продажами.
        Высокое — может быть попыткой создать видимость брендового ассортимента.
        """
        df_copy.loc[:, 'low_variety_count'] = (df_copy['ItemVarietyCount']<2).astype(int)
        df_copy.loc[:, 'high_variety_count'] = (df_copy['ItemVarietyCount']>10).astype(int)
        """
        Если товар недоступен, но есть продажи или возвраты — подозрительно.
        """
        df_copy.loc[:, 'sales_without_availability'] = ((df_copy['ItemAvailableCount']==0)&((df_copy['item_count_sales30']>0) | (df_copy['item_count_returns30']>0))).astype(int)
        """
        Разнообразие товаров у продавца
        Чем больше разнообразных SKU у одного продавца, тем больше шанс, что он просто копирует бренд.
        """
        df_copy.loc[:, 'sku_variety_per_seller'] = df_copy.groupby('SellerID')['ItemVarietyCount'].transform('mean')
        df_copy.loc[:, 'sku_variety_normalized'] = df_copy['ItemVarietyCount']/(df_copy['sku_variety_per_seller']+1)
        """
        Разнообразие против доступности
        Если у товара много вариантов, но почти всё недоступно, это может быть трюк: создать видимость ассортимента, но реально продавать 1-2 позиции.
        """
        df_copy.loc[:, 'variety_to_available_ratio'] = (df_copy['ItemVarietyCount'] / (df_copy['ItemAvailableCount']+1)).clip(upper=20)
        """
        Средняя доступность товара у продавца
        Чтобы понять, торгует ли продавец "реально" или просто заливает витрину.
        """
        df_copy.loc[:, 'avg_available_per_seller'] = df_copy.groupby('SellerID')['ItemAvailableCount'].transform('mean')
        """
        Доля товаров с нулевой доступностью у продавца
        Если у продавца почти все товары недоступны, но есть активность — это подозрительно.
        """
        df_copy.loc[:, 'zero_available_items_per_seller'] = df_copy.groupby('SellerID')['ItemAvailableCount'].transform(lambda x: (x == 0).sum())
        df_copy.loc[:, 'zero_available_ratio'] = df_copy['zero_available_items_per_seller'] / (df_copy['seller_items_count']+1)
        """
        Уникальность товара у продавца — сколько разных SKU в ассортименте
        """
        df_copy.loc[:, 'unique_items_per_seller'] = df_copy.groupby('SellerID')['ItemID'].transform('nunique')
        """
        Доля "псевдо-вариаций" — если у товара много вариаций, но почти все недоступны
        """
        df_copy.loc[:, 'unavailable_ratio_in_variety'] = (df_copy['ItemVarietyCount']-df_copy['ItemAvailableCount'])/(df_copy['ItemVarietyCount']+1)

        df_copy = df_copy.copy()
        return df_copy

    """
    6. Цена
    Цена — мощный сигнал. Подделка часто дёшево.
    
    Фичи:
    PriceDiscounted — ниже ли средней цены по бренду/категории?
    Сравнение с другими товарами той же категории (CommercialTypeName4)
    
    Можно вводить:
    price_ratio_in_category = Price / median_price_in_category
    price_ratio_in_brand = Price / median_price_in_brand
    """
    def feature_engineering_price(self, df_copy):

        df_copy.loc[:, 'price_median_price_category_ratio'] = df_copy['PriceDiscounted']/ (df_copy.groupby('CommercialCategory')['PriceDiscounted'].transform('median')+1)
        df_copy.loc[:, 'price_median_price_brand_ratio'] = df_copy['PriceDiscounted'] / (df_copy.groupby('brand_name')['PriceDiscounted'].transform('median') + 1)
        """
        Добавь бинарные признаки: дешевле ли, чем медиана
        Полезно сделать простые логические признаки
        """
        df_copy.loc[:, 'is_cheaper_than_median_category'] = (df_copy['PriceDiscounted'] < df_copy.groupby('CommercialCategory')['PriceDiscounted'].transform('median')).astype(int)
        df_copy.loc[:, 'is_cheaper_than_median_brand'] = (df_copy['PriceDiscounted'] < df_copy.groupby('brand_name')['PriceDiscounted'].transform('median')).astype(int)
        """
        Добавь нормализацию цены по логарифму
        Цены часто распределены экспоненциально, поэтому логарифм может помочь
        """
        df_copy.loc[:, 'log_price'] = np.log1p(df_copy['PriceDiscounted'])
        """
        Разница между текущей ценой и медианной
        Признак "насколько цена ниже средней" может быть полезен
        """
        df_copy.loc[:, 'price_diff_from_category_median'] = df_copy['PriceDiscounted'] - (df_copy.groupby('CommercialCategory')['PriceDiscounted'].transform('median'))
        df_copy.loc[:, 'price_diff_from_brand_median'] = df_copy['PriceDiscounted'] - (df_copy.groupby('brand_name')['PriceDiscounted'].transform('median'))
        """
        Z-score (отклонение от медианы по категории или бренду)
        Это покажет, насколько цена выделяется относительно других товаров
        Это поможет выявлять товары, резко выбивающиеся по цене.
        """
        df_copy.loc[:, 'price_zscore_category'] = df_copy['price_diff_from_category_median'] / (df_copy.groupby('CommercialCategory')['PriceDiscounted'].transform('std')+1e-5)
        df_copy.loc[:, 'price_zscore_brand'] = df_copy['price_diff_from_brand_median'] / (df_copy.groupby('brand_name')['PriceDiscounted'].transform('std') + 1e-5)

        df_copy = df_copy.copy()
        return df_copy

    """
    7. Агрегации на уровне бренда/продавца
    Контрафакт может быть целиком от определённого продавца или бренда.
    
    Примеры:
    % контрафактных товаров у SellerID или brand_name
    Распределение return_rate по бренду
    Частота появления бренда
    """
    def feature_engineering_brand_seller(self, df_copy):
        """
        Частота появления бренда / продавца в датасете
        """
        df_copy.loc[:, 'frequency_brand'] = df_copy.groupby('brand_name')['ItemID'].transform('count')
        df_copy.loc[:, 'frequency_seller'] = df_copy.groupby('SellerID')['ItemID'].transform('count')
        """
        Средний return_rate по бренду и продавцу
        Если есть return_rate (доля возвратов), это очень полезный сигнал
        """
        df_copy.loc[:, 'return_rate_brand_7'] = df_copy.groupby('brand_name')['item_return_rate_7'].transform('mean')
        df_copy.loc[:, 'return_rate_brand_30'] = df_copy.groupby('brand_name')['item_return_rate_30'].transform('mean')
        df_copy.loc[:, 'return_rate_brand_90'] = df_copy.groupby('brand_name')['item_return_rate_90'].transform('mean')
        df_copy.loc[:, 'return_rate_seller_7'] = df_copy.groupby('SellerID')['item_return_rate_7'].transform('mean')
        df_copy.loc[:, 'return_rate_seller_30'] = df_copy.groupby('SellerID')['item_return_rate_30'].transform('mean')
        df_copy.loc[:, 'return_rate_seller_90'] = df_copy.groupby('SellerID')['item_return_rate_90'].transform('mean')
        """
        Разнообразие товаров у бренда/продавца
        """
        df_copy.loc[:, 'unique_items_per_brand'] = df_copy.groupby('brand_name')['ItemID'].transform('nunique')
        """
        Фичи с отношением количества товаров к уникальности
        Покажут, сколько повторов одного и того же товара (у продавца/бренда)
        """
        df_copy.loc[:, 'repeat_items_ratio_brand'] = df_copy['frequency_brand']/(df_copy['unique_items_per_brand']+1)
        df_copy.loc[:, 'repeat_items_ratio_seller'] = df_copy['frequency_seller'] / (df_copy['unique_items_per_seller'] + 1)
        """
        fake_items_ratio_seller	% контрафакта у продавца
        fake_items_ratio_brand	% контрафакта у бренда
        seller_item_count	Кол-во товаров от продавца
        brand_item_count	Кол-во товаров у бренда
        avg_return_rate_seller	Средний возврат по продавцу
        avg_return_rate_brand	Средний возврат по бренду
        unique_items_per_seller	Разнообразие у продавца
        unique_items_per_brand	Разнообразие у бренда
        """

        df_copy = df_copy.copy()
        return df_copy

    """
    Использование NER и NLP — хорошая идея, но разверни это сильнее
    
    Ты упомянул про NER на базе Natasha. Отлично! Вот как можно это расширить:
    
    🔍 Как использовать NER:
    
    Извлечение брендов, моделей, типов товаров и других сущностей из описаний — даже если они не совпадают с brand_name или CommercialCategory.
    
    Поиск подделок по ключевым словам:
    
    "аналог", "реплика", "репликация", "в стиле", "как оригинал", "100% оригинал" — многие продавцы используют завуалированные формулировки.
    
    Можно собрать словарь таких "red-flag" слов и искать их в тексте.
    
    Сигнал "NER vs заявленные метаданные":
    
    Например: если в тексте описания NER нашёл "Nike", а в brand_name указано что-то нейтральное — подозрительно.
    
    👉 Фичи, которые можно извлечь:
    df['brand_in_description'] = df['description'].apply(lambda x: 1 if 'nike' in x.lower() else 0)
    df['mentions_original'] = df['description'].str.lower().str.contains('оригинал|100% оригинал|настоящий|authentic').astype(int)
    df['mentions_replica'] = df['description'].str.lower().str.contains('реплика|аналог|в стиле|копия').astype(int)
    
    """
    """
    интеграция результатов работы NER-модели в ваш ML пайплайн, то есть:
    
    Обработка текстовых полей (description, name_rus, brand_name)
    Извлечение NER-сущностей
    Преобразование сущностей в числовые фичи
    Добавление их в датасет как новые колонки
    Передача в CatBoost
    """
    """"
    Количество сущностей каждого типа в каждом тексте
    Например, для каждого поля (description, name_rus) посчитать:
    
    Кол-во NAME
    Кол-во PRICE
    Кол-во CATEGORY
    Кол-во BRAND
    Кол-во COUNTRY
    
    Это поможет понять, сколько разных сущностей упоминается и выявить аномалии (например, слишком много или слишком мало брендов).
    """
    def count_entities_by_tag(self, entities, tag):
        return sum(1 for entity in entities if entity['tag'] == tag)
    """
    Флаги  наличия сущностей (булевы признаки)
    Есть ли хоть одна сущность определённого типа?
    Например, есть ли бренд в description, есть ли цена в name_rus и т.д.
    """
    def has_entity_tag(self, entities, tag):
        return any(entity['tag'] == tag for entity in entities)
    """
    Уникальные сущности по типам
    Сохранить уникальные сущности (тексты) для каждого типа и поля — потом можно использовать для сравнения, нормализации и анализа.
    """
    def unique_entities_by_tag(self, entities, tag):
        return list(set(entity['text'] for entity in entities if entity['tag'] == tag))
    """
    Средний и максимальный confidence (уверенность) для каждой сущности по типу
    Низкий confidence может сигнализировать о проблемах распознавания или сомнительных данных.
    """
    def avg_confidence_by_tag(self, entities, tag):
        scores = [entity['score'] for entity in entities if entity['tag'] == tag]
        return sum(scores)/len(scores) if scores else 0

    def max_confidence_by_tag(self, entities, tag):
        scores = [entity['score'] for entity in entities if entity['tag'] == tag]
        return max(scores) if scores else 0

    """
    Сравнение сущностей между полями
    Например, совпадают ли бренды в description и name_rus?
    Совпадение цен? Категорий?
    Это важно для выявления несоответствий, которые могут указывать на подделку.
    """
    def tags_intersaction(self, ent_1, ent_2, tag):
        set_1 = set(entity['text'] for entity in ent_1 if entity['tag'] == tag)
        set_2 = set(entity['text'] for entity in ent_2 if entity['tag'] == tag)
        return len(set_1.intersection(set_2))>0
    """
    Фичи на основе количества разных типов сущностей в одном описании
    Сколько различных типов сущностей найдено в каждом поле? Чем больше разных типов — тем более информативен текст.
    """
    def count_unique_tags(self, entities):
        return len(set(entity['tag'] for entity in entities))
    """
    Добавь дисперсию (variance) по confidence
    Низкая/высокая вариативность может быть важна
    """

    def var_confidence_by_tag(self, entities, tag):
        scores = [entity['score'] for entity in entities if entity['tag'] == tag]
        if len(scores) < 2:
            return 0
        mean = sum(scores) / len(scores)
        return sum((s-mean)**2 for s in scores)/(len(scores) - 1)
    """
    Сравнение значений между PRICE и типичной ценой в категории/бренде
    """
    def extract_price(self, entities):
        for entity in entities:
            if entity['tag'] == 'PRICE':
                try:
                    return float(entity['text'].replace(' ',''))
                except:
                    return None
        return None

    def NER(self, df_copy):
        ner_model = NER_Preprocessing()

        df_copy['ner_description'] = df_copy['description'].apply(lambda x: ner_model.extract_NER_entities(x, verbose=True))
        df_copy['ner_name_rus'] = df_copy['name_rus'].apply(lambda x: ner_model.extract_NER_entities(x, verbose=True))

        for col in ['ner_description', 'ner_name_rus']:
            if col not in df_copy.columns:
                print(f"Колонка {col} отсутствует в df_copy")
                continue  # пропускаем итерацию

            df_copy[col] = df_copy[col].fillna('')

            for tag in ['NAME', 'PRICE', 'CATEGORY', 'BRAND', 'COUNTRY']:
                df_copy[f'{tag.lower()}_count_{col}'] = df_copy[col].apply(lambda ent: self.count_entities_by_tag(ent,tag))
                df_copy[f'unique_{tag.lower()}_count_{col}'] = df_copy[col].apply(lambda ent: len(self.unique_entities_by_tag(ent, tag)))
                df_copy[f'has_{tag.lower()}_{col}'] = df_copy[col].apply(lambda ent: self.has_entity_tag(ent,tag))
                df_copy[f'avg_conf_{tag.lower()}_{col}'] = df_copy[col].apply(lambda ent: self.avg_confidence_by_tag(ent, tag))
                df_copy[f'max_conf_{tag.lower()}_{col}'] = df_copy[col].apply(lambda ent: self.max_confidence_by_tag(ent, tag))
                df_copy[f'var_conf_{tag.lower()}_{col}'] = df_copy[col].apply(lambda ent: self.var_confidence_by_tag(ent, tag))
        for tag in ['NAME', 'PRICE', 'CATEGORY', 'BRAND', 'COUNTRY']:
            df_copy[f'{tag.lower()}_match_desc_name'] = df_copy.apply(lambda ent: self.tags_intersaction(ent['ner_description'], ent['ner_name_rus'], tag), axis=1)

        """
        Фичи на согласованность между разными полями
        Ты уже добавил совпадение сущностей по тегам между description и name_rus. Можно расширить и добавить проверку на несовпадения (например, есть BRAND в description, но нет в name_rus, или наоборот).
        """
        df_copy[f'brand_in_desc_not_in_name'] = df_copy.apply(lambda ent: self.has_entity_tag(ent['ner_description'], 'BRAND') and not self.has_entity_tag(ent['ner_name_rus'], 'BRAND'), axis=1)
        df_copy[f'category_in_name_not_in_desc'] = df_copy.apply( lambda ent: self.has_entity_tag(ent['ner_name_rus'], 'CATEGORY') and not self.has_entity_tag(ent['ner_description'], 'CATEGORY'), axis=1)
        df_copy['unique_tag_count_description'] = df_copy['ner_description'].apply(self.count_unique_tags)
        df_copy['unique_tag_count_name_rus'] = df_copy['ner_name_rus'].apply(self.count_unique_tags)

        """
        Обработка пустых и некорректных данных
        Для стабильности обучения и анализа можно добавить явные фичи с наличием вообще сущностей в тексте.
        """

        df_copy['has_any_entity_description'] = df_copy['ner_description'].apply(lambda ent: len(ent)>0)
        df_copy['has_any_entity_name_rus'] = df_copy['ner_name_rus'].apply(lambda ent: len(ent) > 0)
        df_copy['price_extracted'] = df_copy['ner_description'].apply(lambda ent: self.extract_price(ent))
        df_copy['price_extracted'] = pd.to_numeric(df_copy['price_extracted'], errors='coerce')

        """
        Сравнение цены с эталоном по бренду/категории (ценовые аномалии)
        Если есть колонки brand_name, category_name, то ты можешь
        """
        df_copy['price_extracted'] = df_copy['price_extracted'].fillna(-1)
        brand_median_prices = df_copy.groupby('brand_name')['price_extracted'].median()
        df_copy['price_diff_from_brand_median_ner'] = df_copy.apply(lambda row: abs(row['price_extracted'] - brand_median_prices.get(row['brand_name'], row['price_extracted'])) if row['price_extracted'] else None, axis=1)
        """
        Подозрительные случаи с несколькими BRAND / CATEGORY в одном описании
        """
        df_copy['multiple_brands_in_desc'] = df_copy['ner_description'].apply(lambda ent: self.count_entities_by_tag(ent, 'BRAND') >1)

        return df_copy
"""     
Обогащение внешними источниками

Если доступно:

Проверка брендов по внешним спискам (например, вендор-листы, бренд-реестры, патенты)

Если есть визуальные данные (фото) — можно обучить простую модель или использовать pre-trained CNN (например, CLIP) для поиска похожих изображений.
"""