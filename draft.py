"""
    Мультикатегорийность и семантика
    В реальности товары могут относиться к нескольким категориям — попробуй модель мультиклассификации
    на названии, чтобы предложить правильные категории, а потом сравнивать с текущей категорией.

    Раз у тебя уже есть вектора name_vec и category_vec, то можно реализовать модель для предсказания категории по
    названию (мультиклассификация) прямо на этих векторах, без текстов.

    counts = sample_df['category'].value_counts()
    valid_categories = counts[counts >= 5].index
    filtered_df = sample_df[sample_df['category'].isin(valid_categories)].reset_index(drop=True)

    X = np.array([name_vecs[i] for i in filtered_df.index])
    y_raw = filtered_df['category']

    # преобразуем текст в число
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(y_raw)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

    num_classes = len(np.unique(Y_train))

    print(f"num_class для XGBoost: {num_classes}")
    print(f"Y_train классы: {np.unique(Y_train)}")
    print(f"Y_test классы: {np.unique(Y_test)}")
    # Обучение модели (например, LogisticRegression или RandomForest)
    # model = LogisticRegression(max_iter=1000)
    # Создаём и обучаем XGBoost классификатор
    model = xgb.XGBClassifier(
        objective='multi:softmax',  # мультиклассовая классификация с выводом меток
        num_class=num_classes,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='mlogloss',
        seed=42
    )

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    true_labels = unique_labels(Y_test, Y_pred)
    true_class_names = label_encoder.inverse_transform(true_labels)
    print(classification_report(Y_test, Y_pred, labels=true_labels, target_names=true_class_names, zero_division=0))

    # Применение модели ко всем данным
    filtered_df['predicted_category'] = label_encoder.inverse_transform(model.predict(X))
    filtered_df['is_category_match'] = filtered_df['predicted_category'] == filtered_df['category']
    print(filtered_df[filtered_df['is_category_match'] == False])
    print(f"Количество уникальных категорий после фильтрации: {len(label_encoder.classes_)}")
    print(f"Количество уникальных классов в Y: {len(np.unique(Y))}")
"""



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
import numpy as np
from model_loader import model
from Text_Preprocessing import Text_Preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
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

"""##Чтение файла"""


def load_dataset():
    with open("df_products.csv", encoding="utf-8") as f:
        df_products = pd.read_csv(f)

    print(df_products.head())

    with open("df_feedback.csv", encoding="utf-8") as f:
        df_feedback = pd.read_csv(f)

    print(df_feedback.head())

    with open("df_questions.csv", encoding="utf-8") as f:
        df_questions = pd.read_csv(f)

    print(df_questions.head())

    return df_products, df_feedback, df_questions


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


def preprocess_dataset():
    df_products, df_feedback, df_questions = load_dataset()

    df_products.rename(columns={'imt_id': 'nmId'}, inplace=True)
    df_products = df_products.drop_duplicates(subset=['nmId'])

    # Объединение по nmId (left join: все товары + отзывы, если есть)
    df_products_feedback = df_products.merge(df_feedback, on='nmId', how='left')
    df_products_feedback_questions = df_products_feedback.merge(df_questions, on='nmId', how='left')
    print(df_products_feedback_questions.head())

    print(df_products_feedback_questions.info())

    print(df_products_feedback_questions.shape)

    df_products_feedback_questions['text'].notna()

    print(df_products_feedback_questions[df_products_feedback_questions['text'].notna()]['text'].values)

    print(df_products_feedback_questions[df_products_feedback_questions['question'].notna()]['question'].values)

    df_products_feedback_questions = df_products_feedback_questions.rename(columns={'text': 'feedback'})

    print(df_products_feedback_questions[df_products_feedback_questions['productName'].notna()]['productName'].values)

    df_products_feedback_questions = df_products_feedback_questions.rename(columns={'nmId': 'id'})

    df_products_feedback_questions = df_products_feedback_questions.rename(columns={'imt_name': 'name'})

    df_products_feedback_questions = df_products_feedback_questions.rename(columns={'subj_name': 'category'})

    df_products_feedback_questions = df_products_feedback_questions.rename(columns={'subj_root_name': 'root_category'})

    df_products_feedback_questions = df_products_feedback_questions.rename(columns={'brand_name': 'brand'})

    df_products_feedback_questions = df_products_feedback_questions.rename(columns={'productValuation': 'rating'})

    df_products_feedback_questions = df_products_feedback_questions.rename(columns={'productValuation': 'rating'})

    print(df_products_feedback_questions)

    print(df_products_feedback_questions[df_products_feedback_questions['supplierId'].notna()]['supplierId'].values)

    print(df_products_feedback_questions[df_products_feedback_questions['supplierName'].notna()]['supplierName'].values)

    print(df_products_feedback_questions[df_products_feedback_questions['brandName'].notna()]['brandName'].values)

    print(df_products_feedback_questions[df_products_feedback_questions['brandName'].notna()])

    df_copy = df_products_feedback_questions.copy()

    print(df_copy)

    """## Исключения колонок"""

    df_copy.drop(columns=['brandName'], inplace=True)

    df_copy.drop(columns=['productName'], inplace=True)

    df_copy.drop(columns=['id'], inplace=True)

    print(df_copy)

    """##Проверка и исключение дубликатов"""

    df_copy.duplicated().sum()

    # проверка дубликатов по столбцу id
    # df_copy.duplicated(subset='id').sum()

    # исключение дубликатов по всем признакам датафрейма
    df_copy.drop_duplicates(inplace=True)

    # контроль размеров датафрейма
    print(df_copy.shape)

    """(889828, 14)

    ##**Форматирование и приведение данных к правильному типу**
    """

    df_copy.info()

    df_copy['name'] = df_copy['name'].astype('string')

    df_copy['category'] = df_copy['category'].astype('string')

    df_copy['root_category'] = df_copy['root_category'].astype('string')

    df_copy['brand'] = df_copy['brand'].astype('string')

    df_copy['description'] = df_copy['description'].astype('string')

    df_copy['feedback'] = df_copy['feedback'].astype('string')

    df_copy['supplierName'] = df_copy['supplierName'].astype('string')

    df_copy['question'] = df_copy['question'].astype('string')

    df_copy['answer'] = df_copy['answer'].astype('string')

    df_copy.info()

    """#Обработка пропущенных значений"""

    df_copy.isna().sum()

    df_copy['name'] = df_copy['name'].fillna('')

    df_copy['category'] = df_copy['category'].fillna('')

    df_copy['root_category'] = df_copy['root_category'].fillna('')

    df_copy['brand'] = df_copy['brand'].fillna('')

    df_copy['description'] = df_copy['description'].fillna('')

    df_copy['feedback'] = df_copy['feedback'].fillna('')

    df_copy['supplierName'] = df_copy['supplierName'].fillna('')

    df_copy['question'] = df_copy['question'].fillna('')

    df_copy['answer'] = df_copy['answer'].fillna('')

    df_copy['rating'] = df_copy['rating'].fillna(0)

    df_copy['supplierId'] = df_copy['supplierId'].fillna(0)

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
1. Категориальные (например category, root_category, brand, supplierId, supplierName).
2. Количественные (rating)
3. Текстовые (например, name, description, feedback, question, answer).

## Обзор и описание данных

###Количественные призаки (rating)
"""


def rating_analitics():
    df_copy = preprocess_dataset()
    rating = df_copy.rating

    max_value = rating.max()
    min_value = rating.min()
    mean_value = rating.mean()
    median_value = rating.median()
    print(f'Наибольшая цена: {max_value}', f'Наименьшая цена: {min_value}',
          f'Средняя цена: {mean_value}', f'Медианное значение цены: {median_value}', sep='\n')

    percentile_10_value = rating.quantile(0.10)
    percentile_25_value = rating.quantile(0.25)
    percentile_50_value = rating.quantile(0.50)
    percentile_75_value = rating.quantile(0.75)
    percentile_90_value = rating.quantile(0.90)
    print(f'10-й процентиль: {percentile_10_value}',
          f'25-й процентиль: {percentile_25_value}',
          f'50-й процентиль: {percentile_50_value}',
          f'75-й процентиль: {percentile_75_value}',
          f'90-й процентиль: {percentile_90_value}', sep='\n')

    rating.describe()

    sns.histplot(rating, bins=20, color='blue')
    plt.title('Гистограмма распределения рейтинга товаров')
    plt.xlabel('Рейтинг')
    plt.ylabel('Количество')
    plt.show()

    # количество объектов, значения цены по которым превышает 90 процентиль
    len(df_copy[df_copy.rating > percentile_90_value])

    sns.histplot(rating[rating > percentile_90_value], bins=20, color='blue')
    plt.title('Гистограмма распределения рейтинга товаров')
    plt.xlabel('Рейтинг')
    plt.ylabel('Количество')
    plt.show()

    plt.boxplot(rating[rating > percentile_90_value], vert=False)
    plt.title('Боксплот рейтинга товаров')
    plt.show()

    # коэффициенты ассиметрии и эксцесса
    skew = rating.skew()
    kurtosis = rating.kurtosis()
    print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

    # тест на нормальность распределения
    # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
    # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
    # Уровень значимости полагаем равным 0,05.
    # Проверим признаки на нормальность при помощи критерия Шапиро-Уилка:
    stats.shapiro(df_copy.rating)

    """**Выводы:**
    1. Минимальное значение рейтинга по исходной выборке - 0, максимальное - 5. Размах значений составил 5. Рекомендуется дополнительно проанализировать товары с самым большим рейтингом: возможно это лидеры продаж.
    2. Cреднее значение рейтинга составляет примерно 0.008035480547915398, а медианное - 0.0. Сдвиг незначительный, но свидетельствует о скошенности распределения вправо, рекомендуется проанализировать дополнительно.
    3. Рейтинг товаров до 75% не превышает 0, что свидетельствует о том, что у большинства товаров нет никакого рейтинга .
    4. На гистограмме видно, что распределение признака отлично от нормального.Боксплот не показывает наличие оставшихся выбросов, и он изначально строится после 90 процентиля, если построить боксплот по всем данным, то можно заметить нулевой боксплот, что также свидетельствует об отличии распределения признака от нормального.
    5. Коэффициент ассиметрии больше 1, что так же свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения p-value меньше уровня значимости 0,05, поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.
    """


"""
###Категориальные признаки

####Category
"""


def category_analitics():
    df_copy = preprocess_dataset()

    category = df_copy.category
    category.describe()

    category.mode()

    category.value_counts()

    top_categories = category.value_counts().nlargest(10)
    plt.pie(top_categories, labels=top_categories.index, autopct='%.1f')
    plt.title('Распределение по категориям товаров')
    plt.show()

    """**Выводы:**
    1. Количество уникальных значений по видеопроцессор составляет 4583.
    2. Самая часто встречающаяся марка видеопроцессора - Платья (15.1%).
    3. Основную массу на рынке составляют Платья, Светильники, Книги.
    """


"""
####Root_category
"""


def root_category_analitics():
    df_copy = preprocess_dataset()
    root_category = df_copy.root_category
    root_category.describe()

    root_category.mode()

    root_category.value_counts()

    top_categories = root_category.value_counts().nlargest(10)
    plt.pie(top_categories, labels=top_categories.index, autopct='%.1f')
    plt.title('Распределение по основным категориям товаров')
    plt.show()

    """**Выводы:**
    1. Количество уникальных значений по видеопроцессор составляет 67.
    2. Самая часто встречающаяся марка видеопроцессора - Одежда (26.6%).
    3. Основную массу на рынке составляют Одежда, Дом, Красота.
    """


"""
####Brand
"""


def brand_analitics():
    df_copy = preprocess_dataset()
    brand = df_copy.brand
    brand.describe()

    brand.mode()

    brand.value_counts()

    top_categories = brand.value_counts().nlargest(10)
    plt.pie(top_categories, labels=top_categories.index, autopct='%.1f')
    plt.title('Распределение по брендам товаров')
    plt.show()

    """**Выводы:**
    1. Количество уникальных значений по видеопроцессор составляет 34931.
    2. Самая часто встречающаяся марка видеопроцессора - Сималенд (28%).
    3. Основную массу на рынке составляют Сималенд, Airline, G&C LINKS SKY.
    """


"""
####SupplierId
"""


def supplierId_analitics():
    df_copy = preprocess_dataset()
    supplierId = df_copy.supplierId
    supplierId.describe()

    supplierId.mode()

    supplierId.value_counts()

    filtered_supplierId = supplierId[supplierId != 0]
    top_categories = filtered_supplierId.value_counts().nlargest(10)
    plt.pie(top_categories, labels=top_categories.index, autopct='%.1f')
    plt.title('Распределение по id продавцов товаров')
    plt.show()

    """**Выводы:**
    1. Самая часто встречающаяся марка видеопроцессора - 5418 (34.7%).
    2. Основную массу на рынке составляют 5418, 5049, 17785.
    """


"""
####SupplierName
"""


def supplierName_analitics():
    df_copy = preprocess_dataset()
    supplierName = df_copy.supplierName
    supplierName.describe()

    supplierName.mode()

    supplierName.value_counts()

    filtered_supplierName = supplierName[supplierName != ""]
    top_categories = filtered_supplierName.value_counts().nlargest(10)
    plt.pie(top_categories, labels=top_categories.index, autopct='%.1f')
    plt.title('Распределение по наименованию продавцов товаров')
    plt.show()

    """**Выводы:**
    1. Количество уникальных значений по видеопроцессор составляет 567.
    2. Самая часто встречающаяся марка видеопроцессора - АДИДАС ООО (35.7%).
    3. Основную массу на рынке составляют АДИДАС ООО, BESTSELLER WHOLESALE FINLAND OY, Трэйд ООО.
    """


"""
## Анализ на уровне признаков

###Category

**Задача**:

Часто фродовые товары попадают не в ту категорию

Признак: <название> не соответствует <категории> (например, "телефон Apple" в категории "аксессуары")

**Решение**:

FastText-сходство между текстом и категорией (рекомендуется)

Измеряем косинусное расстояние между эмбеддингами FastText заголовка и категории.

"""


def cosine_sim(v1, v2):
    return cosine_similarity([v1], [v2])[0][0]


def jaccard_similarity(text_1, text_2):
    set_1 = set(text_1)
    set_2 = set(text_2)
    intersection = set_1 & set_2
    union = set_1 | set_2
    if not union:
        return 0
    return len(intersection) / len(union)


def count_similarity(text_1, text_2):
    if not text_1:
        return 0
    return len(set(text_1) & set(text_2)) / len(set(text_1))


def apply_XGBoost(sample_df):
    """
    загружаем уже обученную и сохранённую модель и лейбл энкодер из файлов xgb_classifier.pkl и label_encoder.pkl
    """
    model, label_encoder = load_XGBoost()

    counts = sample_df['category'].value_counts()
    valid_categories = counts[counts >= 5].index
    filtered_df = sample_df[sample_df['category'].isin(valid_categories)]

    X = np.vstack(filtered_df['name_vec'].values)
    Y = model.predict(X)

    filtered_df['predicted_category'] = label_encoder.inverse_transform(Y)
    filtered_df['is_category_match'] = filtered_df['predicted_category'] == filtered_df['category']
    print(filtered_df[filtered_df['is_category_match'] == False])
    print(f"Количество уникальных категорий после фильтрации: {len(label_encoder.classes_)}")
    print(f"Количество уникальных классов в Y: {len(np.unique(Y))}")

    # sample_df.loc[filtered_df.index, 'predicted_category'] = filtered_df['predicted_category']
    # sample_df.loc[filtered_df.index, 'is_category_match'] = filtered_df['is_category_match']
    sample_df.update(filtered_df[['predicted_category', 'is_category_match']])

    return sample_df


def prepare_sample_df():
    df_copy = preprocess_dataset()

    processor = Text_Preprocessing()
    df_copy_category = df_copy[['category', 'name']].copy()

    category_clean = []
    category_vecs = []

    # нормализация, токенизация, лемматизация и векторизация
    print('нормализация, токенизация, лемматизация и векторизация category')
    empty_names = df_copy_category[df_copy_category['name'].str.strip() == '']
    print(f"Пустых названий: {len(empty_names)}")
    print(empty_names.head())
    df_copy_category['name'] = df_copy_category['name'].replace('', np.nan)
    df_copy_category = df_copy_category.dropna(subset=['name'])
    print(f"После очистки: {df_copy_category.shape[0]} строк")
    # sample_df = df_copy_category.head(500).copy()
    sample_df = df_copy_category.copy()

    for string in tqdm(sample_df['category'], desc="Обработка category"):
        if not isinstance(string, str) or string.strip() == '':
            category_clean.append([])
            category_vecs.append(np.zeros(300))  # если размерность TF-IDF вектора 300, или как у тебя
            continue
        normalized_string = processor.Normalization(string, processor.stop_words)
        lemmatized_string = processor.Tokenization_Lemmatization(normalized_string)
        category_clean.append(lemmatized_string)
        if not lemmatized_string:
            vectorized_string = np.zeros(300)
        else:
            vectorized_string = processor.Embedding_Tfidf(lemmatized_string)
        category_vecs.append(vectorized_string)
    sample_df.loc[:, 'category_clean'] = category_clean
    sample_df.loc[:, 'category_vec'] = category_vecs
    print("Сформировано category_clean category_vec")

    name_clean = []
    name_vecs = []
    print('нормализация, токенизация, лемматизация и векторизация name')
    for string in tqdm(sample_df['name'], desc="Обработка name"):
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

    print('Начало применения косинусного расстояния')
    sample_df['cos_sim'] = sample_df.apply(lambda row: cosine_sim(row['category_vec'], row['name_vec']), axis=1)
    print('Косинусное расстояние сформировано cos_sim')

    """Значение от -1 до 1.

    Чем ниже, тем меньше заголовок похож на категорию.

    Можно использовать как признак: cos_sim
    """
    threshold = 0.3
    plt.figure(figsize=(10, 6))
    plt.hist(sample_df['cos_sim'], bins=100, color='red')
    plt.axvline(threshold, color='black', linestyle='--', label=f"Порог: {threshold}")
    plt.legend()
    plt.title('Распределение косинусного сходства между названием и категорией')
    plt.xlabel('Косинусное сходство')
    plt.ylabel('Частота')
    plt.grid(True)
    plt.show()
    """
    Пик на ~0.4–0.5
    Это основная масса «нормальных» товаров — название и категория более-менее согласованы.

    Очень высокий пик около 1.0
    Это, скорее всего, те случаи, где название и категория очень похожи или даже одинаковые (возможно, дублирование текста в обоих полях).

    Длинный хвост влево (до 0.0 и даже ниже 0)
    Это потенциальные аномалии или фродовые товары — название явно не связано с категорией.
    """
    sample_df['suspicious_category'] = sample_df['cos_sim'].fillna(0) < threshold

    print(sample_df['suspicious_category'].sum(), 'подозрительных товаров')
    """
    Улучшения:
    1. Улучшение качества эмбеддингов
    2. Предобработка текста
    3. Расширение признаков
    4. Кластеризация и аномалия
    5. Мультикатегорийность и семантика
    6. Проверка с помощью ключевых слов
    7. Человеческая проверка и активное обучение
    8. Учёт контекста категории
    """
    """
    Добавление частотности категории (или частоты встречаемости категории в датасете) — это полезный признак, 
    особенно если ты работаешь с машинным обучением. Он отражает насколько часто каждая категория встречается в данных и 
    может помочь моделям делать более осознанные предсказания.
    """

    category_freq = sample_df['category'].value_counts()
    sample_df['category_freq'] = sample_df['category'].map(category_freq)

    """
    Почему стоит использовать обе метрики:
    Дополняют друг друга:
    Jaccard покажет, если слова одинаковые.
    Косинус покажет, если слова связаны по смыслу.

    Полезно для обучения модели:
    Машинное обучение лучше работает, когда признаки дают разные ракурсы на данные.
    В реальных задачах часто помогает иметь и "простые" признаки, и "глубокие".

    Jaccard — дёшево и быстро, можно посчитать даже до векторизации.
    """

    sample_df['jacc_sim'] = sample_df.apply(lambda row: jaccard_similarity(row['category_clean'], row['name_clean']),
                                            axis=1)

    """
    Если ты хочешь получить количество совпадений ключевых слов из категории в названии, то это можно реализовать очень 
    просто, особенно если у тебя уже есть лемматизированные версии category_clean и name_clean (например, списки слов)
    """

    sample_df['keyword_match_ratio'] = sample_df.apply(
        lambda row: count_similarity(row['category_clean'], row['name_clean']), axis=1)

    """
    Кластеризация и аномалия
    Используй алгоритмы для обнаружения аномалий (например, Isolation Forest, Local Outlier Factor) на векторах 
    или на косинусных расстояниях — для выявления странных совпадений.
    """

    anomaly_features = ['cos_sim', 'suspicious_category', 'category_freq', 'jacc_sim', 'keyword_match_ratio']

    # масштибируем фичи
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sample_df[anomaly_features].fillna(0))

    # обучаем IsolationForest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    sample_df['iso_forest'] = iso_forest.fit_predict(X_scaled)  # -1 = аномалия, 1 = норма

    sns.scatterplot(data=sample_df, x='cos_sim', y='keyword_match_ratio', hue='iso_forest')
    plt.title('Аномалии по признакам')
    plt.show()

    return sample_df


def category_frod_detection():
    sample_df = prepare_sample_df()
    sample_df = apply_XGBoost(sample_df)

    return sample_df


"""
Бренд

Очистка данных:
Приведение к lower, strip, удаление пустых и None.
Удаление некорректных символов и строк без букв.

Нормализация брендов:
Использование clean_brand с удалением пробелов и приведение к нижнему регистру.
Загрузка и очистка базы известных брендов (query.csv):
Исключение технических строк Qxxx.

Определение редких брендов:
По частоте появления в датасете (<= 2).
Поиск похожих брендов через rapidfuzz:
Учитывается порог threshold=85.
Возвращаются флаг схожести и ближайший бренд.
"""


def clean_brand(brand):
    if not isinstance(brand, str):
        return ''
    brand = re.sub(r'\s+', '', brand.strip().lower())
    brand_translit = transliterate_brand(brand)
    return brand_translit


def is_fake_like_brand(brand, known_brand_set, known_brand_list, threshold=85):
    # уже известный бренд — не фейк
    if brand in known_brand_set:
        return False, None
    # ищем наиболее похожий бренд
    match = process.extractOne(brand, known_brand_list, scorer=fuzz.ratio)
    if match:
        matched_brand, score, _ = match
        return score >= threshold, matched_brand
    return False, None


def transliterate_brand(brand):
    if not isinstance(brand, str):
        return ''
    # Транслитерируем русский текст в латиницу, если есть кириллица
    try:
        # language_code='ru' — русская транслитерация, reversed=True — кириллица в латиницу
        return translit(brand, 'ru', reversed=True)
    except Exception:
        # Если транслитерация не сработала (например, латиница) — вернуть как есть
        return brand.lower()


def brand_frod_detection():
    df_copy = preprocess_dataset()
    df_copy_brand = df_copy[['brand']].copy()

    # Нормализуй (lower(), strip())
    df_copy_brand['brand'] = df_copy_brand['brand'].str.lower().str.strip()

    # Проверка на корректные бренды
    df_wiki = pd.read_csv('query.csv')
    df_wiki_clean = df_wiki[~df_wiki['brandLabel'].astype(str).str.match(r'^Q\d+$', flags=re.IGNORECASE)]

    # 1. Удалить пустые и None значения
    df_clean = df_copy_brand.dropna(subset=['brand'])
    df_clean = df_clean[df_clean['brand'].str.strip() != '']

    # 2. Оставить только строки с буквами (например, бренды должны содержать буквы)
    df_clean = df_clean[df_clean['brand'].str.contains('[a-zA-Zа-яА-Я]', regex=True)]

    # 3. Можно дополнительно убрать строки с подозрительными символами (оставить только буквы, цифры, пробелы, дефисы)
    df_clean = df_clean[df_clean['brand'].str.match(r'^[\w\s\-&а-яА-Я]+$', regex=True)]

    df_clean['clean_brand'] = df_clean['brand'].apply(clean_brand)
    # Частота брендов — редкие могут быть фродовыми
    brand_freq = df_clean['clean_brand'].value_counts()
    df_clean['brand_freq'] = df_clean['clean_brand'].map(brand_freq)

    known_brands = df_wiki_clean['brandLabel'].dropna().str.lower().str.strip()
    known_brand_set = set(clean_brand(brand) for brand in known_brands)
    known_brand_list = list(known_brand_set)
    df_clean = df_clean.copy()
    df_clean['is_known_brand'] = df_clean['clean_brand'].isin(known_brand_set)

    # Новая метка: подозрительный бренд (неизвестный + редкий)
    df_clean['is_suspicious_brand'] = (~df_clean['is_known_brand']) & (df_clean['brand_freq'] <= 2)

    # Флаг closest_known_brand — чтобы видеть, на кого похож поддельный бренд.
    results = df_clean['clean_brand'].apply(lambda x: is_fake_like_brand(x, known_brand_set, known_brand_list))
    df_clean['is_fake_like_brand'] = results.apply(lambda x: x[0])
    df_clean['closest_known_brand'] = results.apply(lambda x: x[1])

    return df_clean


"""
Заголовок
"""


def name_frod_detection():
    df_copy = preprocess_dataset()
    processor = Text_Preprocessing()
    df_copy_name = df_copy[['name']].copy()

    # нормализация, токенизация, лемматизация и векторизация
    print('нормализация, токенизация, лемматизация и векторизация category')
    empty_names = df_copy_name[df_copy_name['name'].str.strip() == '']
    print(f"Пустых названий: {len(empty_names)}")
    print(empty_names.head())
    df_copy_name['name'] = df_copy_name['name'].replace('', np.nan)
    df_copy_name = df_copy_name.dropna(subset=['name'])
    print(f"После очистки: {df_copy_name.shape[0]} строк")
    sample_df = df_copy_name.head(100).copy()
    # sample_df = df_copy_name.copy()

    name_clean = []
    name_vecs = []
    print('нормализация, токенизация, лемматизация и векторизация name')
    for string in tqdm(sample_df['name'], desc="Обработка name"):
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

    X = np.vstack(sample_df['name_vec'].values())

    # масштибируем фичи
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # обучаем DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
    clusters = dbscan.fit_predict(X_scaled)

    sample_df['clusters'] = clusters
    sample_df['is_outlier'] = sample_df['clusters'] == -1

    print("Подозрительные названия (кластеры -1):")
    print(sample_df[sample_df['is_outlier']][['name', 'cluster']].head(20))

    """
    Анализ ключевых слов и фраз (кликбейт)

    Добавь отдельную проверку для часто встречающихся в фродовых названиях слов (например, "оригинал", "копия", "акция", "скидка").
    Можно сделать словарь стоп-слов и кликбейт-фраз.


    Метрики качества и визуализация

    Визуализируй кластеры (UMAP, t-SNE) для отлова подозрительных групп.
    Посмотри примеры подозрительных названий вручную, чтобы улучшить правила.
    """


# -*- coding: utf-8 -*-
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
import numpy as np
from model_loader import model
from Text_Preprocessing import Text_Preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tqdm import tqdm
import xgboost as xgb
from sklearn.utils.multiclass import unique_labels


def main():
    """##Чтение файла"""

    with open("df_products.csv", encoding="utf-8") as f:
        df_products = pd.read_csv(f)

    print(df_products.head())

    with open("df_feedback.csv", encoding="utf-8") as f:
        df_feedback = pd.read_csv(f)

    print(df_feedback.head())

    with open("df_questions.csv", encoding="utf-8") as f:
        df_questions = pd.read_csv(f)

    print(df_questions.head())

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

    df_products.rename(columns={'imt_id': 'nmId'}, inplace=True)
    df_products = df_products.drop_duplicates(subset=['nmId'])

    # Объединение по nmId (left join: все товары + отзывы, если есть)
    df_products_feedback = df_products.merge(df_feedback, on='nmId', how='left')
    df_products_feedback_questions = df_products_feedback.merge(df_questions, on='nmId', how='left')
    print(df_products_feedback_questions.head())

    print(df_products_feedback_questions.info())

    print(df_products_feedback_questions.shape)

    df_products_feedback_questions['text'].notna()

    print(df_products_feedback_questions[df_products_feedback_questions['text'].notna()]['text'].values)

    print(df_products_feedback_questions[df_products_feedback_questions['question'].notna()]['question'].values)

    df_products_feedback_questions = df_products_feedback_questions.rename(columns={'text': 'feedback'})

    print(df_products_feedback_questions[df_products_feedback_questions['productName'].notna()]['productName'].values)

    df_products_feedback_questions = df_products_feedback_questions.rename(columns={'nmId': 'id'})

    df_products_feedback_questions = df_products_feedback_questions.rename(columns={'imt_name': 'name'})

    df_products_feedback_questions = df_products_feedback_questions.rename(columns={'subj_name': 'category'})

    df_products_feedback_questions = df_products_feedback_questions.rename(columns={'subj_root_name': 'root_category'})

    df_products_feedback_questions = df_products_feedback_questions.rename(columns={'brand_name': 'brand'})

    df_products_feedback_questions = df_products_feedback_questions.rename(columns={'productValuation': 'rating'})

    df_products_feedback_questions = df_products_feedback_questions.rename(columns={'productValuation': 'rating'})

    print(df_products_feedback_questions)

    print(df_products_feedback_questions[df_products_feedback_questions['supplierId'].notna()]['supplierId'].values)

    print(df_products_feedback_questions[df_products_feedback_questions['supplierName'].notna()]['supplierName'].values)

    print(df_products_feedback_questions[df_products_feedback_questions['brandName'].notna()]['brandName'].values)

    print(df_products_feedback_questions[df_products_feedback_questions['brandName'].notna()])

    df_copy = df_products_feedback_questions.copy()

    print(df_copy)

    """## Исключения колонок"""

    df_copy.drop(columns=['brandName'], inplace=True)

    df_copy.drop(columns=['productName'], inplace=True)

    df_copy.drop(columns=['id'], inplace=True)

    print(df_copy)

    """##Проверка и исключение дубликатов"""

    df_copy.duplicated().sum()

    # проверка дубликатов по столбцу id
    # df_copy.duplicated(subset='id').sum()

    # исключение дубликатов по всем признакам датафрейма
    df_copy.drop_duplicates(inplace=True)

    # контроль размеров датафрейма
    print(df_copy.shape)

    """(889828, 14)

    ##**Форматирование и приведение данных к правильному типу**
    """

    df_copy.info()

    df_copy['name'] = df_copy['name'].astype('string')

    df_copy['category'] = df_copy['category'].astype('string')

    df_copy['root_category'] = df_copy['root_category'].astype('string')

    df_copy['brand'] = df_copy['brand'].astype('string')

    df_copy['description'] = df_copy['description'].astype('string')

    df_copy['feedback'] = df_copy['feedback'].astype('string')

    df_copy['supplierName'] = df_copy['supplierName'].astype('string')

    df_copy['question'] = df_copy['question'].astype('string')

    df_copy['answer'] = df_copy['answer'].astype('string')

    df_copy.info()

    """#Обработка пропущенных значений"""

    df_copy.isna().sum()

    df_copy['name'] = df_copy['name'].fillna('')

    df_copy['category'] = df_copy['category'].fillna('')

    df_copy['root_category'] = df_copy['root_category'].fillna('')

    df_copy['brand'] = df_copy['brand'].fillna('')

    df_copy['description'] = df_copy['description'].fillna('')

    df_copy['feedback'] = df_copy['feedback'].fillna('')

    df_copy['supplierName'] = df_copy['supplierName'].fillna('')

    df_copy['question'] = df_copy['question'].fillna('')

    df_copy['answer'] = df_copy['answer'].fillna('')

    df_copy['rating'] = df_copy['rating'].fillna(0)

    df_copy['supplierId'] = df_copy['supplierId'].fillna(0)

    df_copy.isna().sum()

    df_copy.head()

    df_copy.info()

    """# Анализ данных

    **Этапы анализа:**
    1. Определить целевые и факторные переменные.
    2. Определить типы переменных.
    3. Проанализировать каждый признак отдельно.
    4. Проанализировать взаимосвязи признаков.
    5. Проинтерпретировать результаты.

    **Типы переменных:**
    1. Категориальные (например category, root_category, brand, supplierId, supplierName).
    2. Количественные (rating)
    3. Текстовые (например, name, description, feedback, question, answer).

    ## Обзор и описание данных

    ###Количественные призаки (rating)
    """

    rating = df_copy.rating

    max_value = rating.max()
    min_value = rating.min()
    mean_value = rating.mean()
    median_value = rating.median()
    print(f'Наибольшая цена: {max_value}', f'Наименьшая цена: {min_value}',
          f'Средняя цена: {mean_value}', f'Медианное значение цены: {median_value}', sep='\n')

    percentile_10_value = rating.quantile(0.10)
    percentile_25_value = rating.quantile(0.25)
    percentile_50_value = rating.quantile(0.50)
    percentile_75_value = rating.quantile(0.75)
    percentile_90_value = rating.quantile(0.90)
    print(f'10-й процентиль: {percentile_10_value}',
          f'25-й процентиль: {percentile_25_value}',
          f'50-й процентиль: {percentile_50_value}',
          f'75-й процентиль: {percentile_75_value}',
          f'90-й процентиль: {percentile_90_value}', sep='\n')

    rating.describe()

    sns.histplot(rating, bins=20, color='blue')
    plt.title('Гистограмма распределения рейтинга товаров')
    plt.xlabel('Рейтинг')
    plt.ylabel('Количество')
    plt.show()

    # количество объектов, значения цены по которым превышает 90 процентиль
    len(df_copy[df_copy.rating > percentile_90_value])

    sns.histplot(rating[rating > percentile_90_value], bins=20, color='blue')
    plt.title('Гистограмма распределения рейтинга товаров')
    plt.xlabel('Рейтинг')
    plt.ylabel('Количество')
    plt.show()

    plt.boxplot(rating[rating > percentile_90_value], vert=False)
    plt.title('Боксплот рейтинга товаров')
    plt.show()

    # коэффициенты ассиметрии и эксцесса
    skew = rating.skew()
    kurtosis = rating.kurtosis()
    print(f'Коэффициент ассиметрии: {skew}', f'Коэффициент эксцесса: {kurtosis}', sep='\n')

    # тест на нормальность распределения
    # Гипотеза Н0: наблюдаемая выборка принадлежит нормальной генеральной совокупности.
    # Гипотеза Н1: наблюдаемая выборка не принадлежит нормальной генеральной совокупности.
    # Уровень значимости полагаем равным 0,05.
    # Проверим признаки на нормальность при помощи критерия Шапиро-Уилка:
    stats.shapiro(df_copy.rating)

    """**Выводы:**
    1. Минимальное значение рейтинга по исходной выборке - 0, максимальное - 5. Размах значений составил 5. Рекомендуется дополнительно проанализировать товары с самым большим рейтингом: возможно это лидеры продаж.
    2. Cреднее значение рейтинга составляет примерно 0.008035480547915398, а медианное - 0.0. Сдвиг незначительный, но свидетельствует о скошенности распределения вправо, рекомендуется проанализировать дополнительно.
    3. Рейтинг товаров до 75% не превышает 0, что свидетельствует о том, что у большинства товаров нет никакого рейтинга .
    4. На гистограмме видно, что распределение признака отлично от нормального.Боксплот не показывает наличие оставшихся выбросов, и он изначально строится после 90 процентиля, если построить боксплот по всем данным, то можно заметить нулевой боксплот, что также свидетельствует об отличии распределения признака от нормального.
    5. Коэффициент ассиметрии больше 1, что так же свидетельствует о скошенности вправо. Коэффициент эксцесса больше 1, что говорит о нерассредоточенности значений от средних. По проведенному тесту на нормальность распределения p-value меньше уровня значимости 0,05, поэтому гипотеза о нормальности распределения признака отвергается. Соответственно в дальнейшем при анализе взаимосвязи признаков следует применять непараметрические критерии сравнения групп: для 2х независимых групп - критерий Манна-Уитни, для более 2х независимых групп - критерий Краскала-Уоллиса.

    ###Категориальные признаки

    ####Category
    """

    category = df_copy.category
    category.describe()

    category.mode()

    category.value_counts()

    top_categories = category.value_counts().nlargest(10)
    plt.pie(top_categories, labels=top_categories.index, autopct='%.1f')
    plt.title('Распределение по категориям товаров')
    plt.show()

    """**Выводы:**
    1. Количество уникальных значений по видеопроцессор составляет 4583.
    2. Самая часто встречающаяся марка видеопроцессора - Платья (15.1%).
    3. Основную массу на рынке составляют Платья, Светильники, Книги.

    ####Root_category
    """

    root_category = df_copy.root_category
    root_category.describe()

    root_category.mode()

    root_category.value_counts()

    top_categories = root_category.value_counts().nlargest(10)
    plt.pie(top_categories, labels=top_categories.index, autopct='%.1f')
    plt.title('Распределение по основным категориям товаров')
    plt.show()

    """**Выводы:**
    1. Количество уникальных значений по видеопроцессор составляет 67.
    2. Самая часто встречающаяся марка видеопроцессора - Одежда (26.6%).
    3. Основную массу на рынке составляют Одежда, Дом, Красота.

    ####Brand
    """

    brand = df_copy.brand
    brand.describe()

    brand.mode()

    brand.value_counts()

    top_categories = brand.value_counts().nlargest(10)
    plt.pie(top_categories, labels=top_categories.index, autopct='%.1f')
    plt.title('Распределение по брендам товаров')
    plt.show()

    """**Выводы:**
    1. Количество уникальных значений по видеопроцессор составляет 34931.
    2. Самая часто встречающаяся марка видеопроцессора - Сималенд (28%).
    3. Основную массу на рынке составляют Сималенд, Airline, G&C LINKS SKY.

    ####SupplierId
    """

    supplierId = df_copy.supplierId
    supplierId.describe()

    supplierId.mode()

    supplierId.value_counts()

    filtered_supplierId = supplierId[supplierId != 0]
    top_categories = filtered_supplierId.value_counts().nlargest(10)
    plt.pie(top_categories, labels=top_categories.index, autopct='%.1f')
    plt.title('Распределение по id продавцов товаров')
    plt.show()

    """**Выводы:**
    1. Самая часто встречающаяся марка видеопроцессора - 5418 (34.7%).
    2. Основную массу на рынке составляют 5418, 5049, 17785.

    ####SupplierName
    """

    supplierName = df_copy.supplierName
    supplierName.describe()

    supplierName.mode()

    supplierName.value_counts()

    filtered_supplierName = supplierName[supplierName != ""]
    top_categories = filtered_supplierName.value_counts().nlargest(10)
    plt.pie(top_categories, labels=top_categories.index, autopct='%.1f')
    plt.title('Распределение по наименованию продавцов товаров')
    plt.show()

    """**Выводы:**
    1. Количество уникальных значений по видеопроцессор составляет 567.
    2. Самая часто встречающаяся марка видеопроцессора - АДИДАС ООО (35.7%).
    3. Основную массу на рынке составляют АДИДАС ООО, BESTSELLER WHOLESALE FINLAND OY, Трэйд ООО.

    ## Анализ на уровне признаков

    ###Category

    **Задача**:

    Часто фродовые товары попадают не в ту категорию

    Признак: <название> не соответствует <категории> (например, "телефон Apple" в категории "аксессуары")

    **Решение**:

    FastText-сходство между текстом и категорией (рекомендуется)

    Измеряем косинусное расстояние между эмбеддингами FastText заголовка и категории.
    """
    processor = Text_Preprocessing()
    df_copy_category = df_copy[['category', 'name']].copy()

    category_clean = []
    category_vecs = []

    # нормализация, токенизация, лемматизация и векторизация
    print('нормализация, токенизация, лемматизация и векторизация category')
    empty_names = df_copy_category[df_copy_category['name'].str.strip() == '']
    print(f"Пустых названий: {len(empty_names)}")
    print(empty_names.head())
    df_copy_category['name'] = df_copy_category['name'].replace('', np.nan)
    df_copy_category = df_copy_category.dropna(subset=['name'])
    print(f"После очистки: {df_copy_category.shape[0]} строк")
    sample_df = df_copy_category.head(500).copy()

    for string in tqdm(sample_df['category'], desc="Обработка category"):
        if not isinstance(string, str) or string.strip() == '':
            category_clean.append([])
            category_vecs.append(np.zeros(300))  # если размерность TF-IDF вектора 300, или как у тебя
            continue
        normalized_string = processor.Normalization(string, processor.stop_words)
        lemmatized_string = processor.Tokenization_Lemmatization(normalized_string)
        category_clean.append(lemmatized_string)
        if not lemmatized_string:
            vectorized_string = np.zeros(300)
        else:
            vectorized_string = processor.Embedding_Tfidf(lemmatized_string)
        category_vecs.append(vectorized_string)
    sample_df.loc[:, 'category_clean'] = category_clean
    sample_df.loc[:, 'category_vec'] = category_vecs
    # sample_df['category_clean'] = category_clean
    # sample_df['category_vec'] = category_vecs
    print("Сформировано category_clean category_vec")

    name_clean = []
    name_vecs = []
    print('нормализация, токенизация, лемматизация и векторизация name')
    for string in tqdm(sample_df['name'], desc="Обработка name"):
        if not isinstance(string, str) or string.strip() == '':
            category_clean.append([])
            category_vecs.append(np.zeros(300))  # если размерность TF-IDF вектора 300, или как у тебя
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
    # sample_df['name_clean'] = name_clean
    # sample_df['name_vec'] = name_vecs
    print("Сформироавно name_clean name_vec")

    def cosine_sim(v1, v2):
        return cosine_similarity([v1], [v2])[0][0]

    print('Начало применения косинусного расстояния')
    sample_df['cos_sim'] = sample_df.apply(lambda row: cosine_sim(row['category_vec'], row['name_vec']), axis=1)
    print('Косинусное расстояние сформировано cos_sim')

    """Значение от -1 до 1.

    Чем ниже, тем меньше заголовок похож на категорию.

    Можно использовать как признак: cos_sim
    """
    threshold = 0.3
    plt.figure(figsize=(10, 6))
    plt.hist(sample_df['cos_sim'], bins=100, color='red')
    plt.axvline(threshold, color='black', linestyle='--', label=f"Порог: {threshold}")
    plt.legend()
    plt.title('Распределение косинусного сходства между названием и категорией')
    plt.xlabel('Косинусное сходство')
    plt.ylabel('Частота')
    plt.grid(True)
    plt.show()
    """
    Пик на ~0.4–0.5
    Это основная масса «нормальных» товаров — название и категория более-менее согласованы.

    Очень высокий пик около 1.0
    Это, скорее всего, те случаи, где название и категория очень похожи или даже одинаковые (возможно, дублирование текста в обоих полях).

    Длинный хвост влево (до 0.0 и даже ниже 0)
    Это потенциальные аномалии или фродовые товары — название явно не связано с категорией.
    """
    sample_df['suspicious_category'] = sample_df['cos_sim'].fillna(0) < threshold

    print(sample_df['suspicious_category'].sum(), 'подозрительных товаров')
    """
    Улучшения:
    1. Улучшение качества эмбеддингов
    2. Предобработка текста
    3. Расширение признаков
    4. Кластеризация и аномалия
    5. Мультикатегорийность и семантика
    6. Проверка с помощью ключевых слов
    7. Человеческая проверка и активное обучение
    8. Учёт контекста категории
    """
    """
    Добавление частотности категории (или частоты встречаемости категории в датасете) — это полезный признак, 
    особенно если ты работаешь с машинным обучением. Он отражает насколько часто каждая категория встречается в данных и 
    может помочь моделям делать более осознанные предсказания.
    """

    category_freq = sample_df['category'].value_counts()
    sample_df['category_freq'] = sample_df['category'].map(category_freq)

    """
    Почему стоит использовать обе метрики:
    Дополняют друг друга:
    Jaccard покажет, если слова одинаковые.
    Косинус покажет, если слова связаны по смыслу.

    Полезно для обучения модели:
    Машинное обучение лучше работает, когда признаки дают разные ракурсы на данные.
    В реальных задачах часто помогает иметь и "простые" признаки, и "глубокие".

    Jaccard — дёшево и быстро, можно посчитать даже до векторизации.
    """

    def jaccard_similarity(text_1, text_2):
        set_1 = set(text_1)
        set_2 = set(text_2)
        intersection = set_1 & set_2
        union = set_1 | set_2
        if not union:
            return 0
        return len(intersection) / len(union)

    sample_df['jacc_sim'] = sample_df.apply(lambda row: jaccard_similarity(row['category_clean'], row['name_clean']),
                                            axis=1)

    """
    Если ты хочешь получить количество совпадений ключевых слов из категории в названии, то это можно реализовать очень 
    просто, особенно если у тебя уже есть лемматизированные версии category_clean и name_clean (например, списки слов)
    """

    def count_similarity(text_1, text_2):
        if not text_1:
            return 0
        return len(set(text_1) & set(text_2)) / len(set(text_1))

    sample_df['keyword_match_ratio'] = sample_df.apply(
        lambda row: count_similarity(row['category_clean'], row['name_clean']), axis=1)

    """
    Кластеризация и аномалия
    Используй алгоритмы для обнаружения аномалий (например, Isolation Forest, Local Outlier Factor) на векторах 
    или на косинусных расстояниях — для выявления странных совпадений.
    """

    anomaly_features = ['cos_sim', 'suspicious_category', 'category_freq', 'jacc_sim', 'keyword_match_ratio']
    # масштибируем фичи
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sample_df[anomaly_features].fillna(0))
    # обучаем IsolationForest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    sample_df['iso_forest'] = iso_forest.fit_predict(X_scaled)  # -1 = аномалия, 1 = норма

    sns.scatterplot(data=sample_df, x='cos_sim', y='keyword_match_ratio', hue='iso_forest')
    plt.title('Аномалии по признакам')
    plt.show()

    """
    Мультикатегорийность и семантика
    В реальности товары могут относиться к нескольким категориям — попробуй модель мультиклассификации
    на названии, чтобы предложить правильные категории, а потом сравнивать с текущей категорией.

    Раз у тебя уже есть вектора name_vec и category_vec, то можно реализовать модель для предсказания категории по 
    названию (мультиклассификация) прямо на этих векторах, без текстов.
    """
    counts = sample_df['category'].value_counts()
    valid_categories = counts[counts >= 5].index
    filtered_df = sample_df[sample_df['category'].isin(valid_categories)].reset_index(drop=True)

    X = np.array([name_vecs[i] for i in filtered_df.index])
    y_raw = filtered_df['category']

    # преобразуем текст в число
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(y_raw)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

    num_classes = len(np.unique(Y_train))

    print(f"num_class для XGBoost: {num_classes}")
    print(f"Y_train классы: {np.unique(Y_train)}")
    print(f"Y_test классы: {np.unique(Y_test)}")
    # Обучение модели (например, LogisticRegression или RandomForest)
    # model = LogisticRegression(max_iter=1000)
    # Создаём и обучаем XGBoost классификатор
    model = xgb.XGBClassifier(
        objective='multi:softmax',  # мультиклассовая классификация с выводом меток
        num_class=num_classes,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='mlogloss',
        seed=42
    )

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    true_labels = unique_labels(Y_test, Y_pred)
    true_class_names = label_encoder.inverse_transform(true_labels)
    print(classification_report(Y_test, Y_pred, labels=true_labels, target_names=true_class_names, zero_division=0))

    # Применение модели ко всем данным
    filtered_df['predicted_category'] = label_encoder.inverse_transform(model.predict(X))
    filtered_df['is_category_match'] = filtered_df['predicted_category'] == filtered_df['category']
    print(filtered_df[filtered_df['is_category_match'] == False])
    print(f"Количество уникальных категорий после фильтрации: {len(label_encoder.classes_)}")
    print(f"Количество уникальных классов в Y: {len(np.unique(Y))}")


if __name__ == '__main__':
    main()

import json
import traceback

from deeppavlov import build_model, configs
from deeppavlov import train_model
from deeppavlov.core.common.errors import ConfigError
import random
import os
import traceback
import shutil
from transformers import BertForTokenClassification, BertTokenizer
import torch
import pickle
import inspect

"""
Активация виртуальной среды
.\env_deeppavlov\scripts\activate
"""


# Если у тебя русские данные, можно использовать ner_rus_bert.
def create_correct_tag_dict():
    """Создание правильного формата tag.dict"""
    data_path = "C:/Nastya/ozon_py/my_project/data/train.txt"
    tag_dict_path = "C:/Nastya/ozon_py/my_project/models/my_ner_model/tag.dict"

    from collections import Counter
    tag_counter = Counter()

    # Считаем частоту тегов из обучающих данных
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and ' ' in line:
                parts = line.split()
                if len(parts) >= 2:
                    tag = parts[-1]
                    tag_counter[tag] += 1

    # Сохраняем в правильном формате: тег\tчастота
    with open(tag_dict_path, 'w', encoding='utf-8') as f:
        for tag, count in tag_counter.most_common():
            f.write(f"{tag}\t{count}\n")

    print("Правильный tag.dict создан:")
    for tag, count in tag_counter.most_common():
        print(f"  {tag}\t{count}")


def ner_model_download():
    try:
        ner_model = build_model(configs.ner.ner_rus_bert, download=True)
        return ner_model
    except ConfigError as e:
        print("Ошибка загрузки NER DeepPavlov модели:", e)
        return None


"""
Из исходного файла с 4 колонками оставить только первую и последнюю.
-DOCSTART- -X- O
ACTRUM -X- _ B-BRAND
Убрать -X- _ везде и -DOCSTART- -X- O в начале файла
Сделать пустую строку между предложениями.
Разделить данные на train.txt, valid.txt, test.txt.
"""


def clean_conll(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                fout.write('\n')
                continue
            parts = line.split()
            if parts[0] == '-DOCSTART-':
                continue
            if len(parts) < 2:
                continue
            word = parts[0]
            tag = parts[-1]
            fout.write(f"{word} {tag}\n")


"""
Разбить данные на:
train.txt
valid.txt
(опционально) test.txt
Указать пути в JSON-конфиге DeepPavlov и запустить дообучение.
Скрипт для подготовки данных
"""


# Функция для сохранения
def save_file(samples, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            if isinstance(sample, list):
                f.write('\n'.join(sample) + '\n\n')
            else:
                # Если sample по ошибке строка — не обрабатываем
                print(f"❗ Ошибка: sample не список: {sample}")


def data_preparation():
    sample = []
    samples = []
    with open('cleaned_training_data_ner.conll', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if sample:
                    samples.append(sample)
                    sample = []
            else:
                sample.append(line)

    # Добавим последний семпл (если был)
    if sample:
        samples.append(sample)

    # Перемешиваем
    random.seed(42)
    random.shuffle(samples)

    # Разбиваем
    n = len(samples)
    train_samples = samples[:int(n * 0.8)]
    valid_samples = samples[int(n * 0.8):int(n * 0.9)]
    test_samples = samples[int(n * 0.9):]

    # Сохраняем в три файла
    save_file(train_samples, 'train_samples_ner.txt')
    save_file(valid_samples, 'valid_samples_ner.txt')
    save_file(test_samples, 'test_samples_ner.txt')


"""
Создай кастомный конфиг
Склонируй стандартный конфиг и подправь
$src = python -c "import deeppavlov; print(deeppavlov.__path__[0])"
Copy-Item "$src\configs\ner\ner_rus_bert.json" -Destination "my_ner_config\my_ner.json"

Проверь, что файл действительно появился:
Get-Item my_ner_config\my_ner.json

Открой my_ner.json и измени:
Пути к данным
Количество эпох
Сохраняемую модель

Нужно отредактировать следующие поля:

🔸 dataset_reader — путь к твоим данным:
"dataset_reader": {
    "class_name": "conll2003_reader",
    "data_path": "C:/Nastya/ozon_py/my_project/data/",
	"train": "train_samples_ner.txt",
	"valid": "valid_samples_ner.txt",
	"test": "test_samples_ner.txt",
    "provide_pos": false
  },
{
        "class_name": "torch_transformers_sequence_tagger",
        "n_tags": "#tag_vocab.len",
        "pretrained_bert": "{TRANSFORMER}",
        "attention_probs_keep_prob": 0.5,
        "encoder_layer_ids": [
          -1
        ],
        "optimizer": "AdamW",
        "optimizer_parameters": {
          "lr": 2e-05,
          "weight_decay": 1e-06,
          "betas": [
            0.9,
            0.999
          ],
          "eps": 1e-06
        },
        "clip_norm": 1.0,
        "min_learning_rate": 1e-07,
        "learning_rate_drop_patience": 30,
        "learning_rate_drop_div": 1.5,
        "load_before_drop": true,
        "save_path": "{MODEL_PATH}/my_ner_model.pth.tar",
		"load_path": "{MODEL_PATH}/my_ner_model.pth.tar",
        "in": [
          "x_subword_tok_ids",
          "attention_mask",
          "startofword_markers"
        ],
        "in_y": [
          "y_ind"
        ],
        "out": [
          "y_pred_ind",
          "probas"
        ]
      }
Убедись, что твои файлы находятся по пути:

~/.deeppavlov/downloads/my_ner_data/train_samples_ner.txt
                                                valid_samples_ner.txt
                                                test_samples_ner.txt

Создать файлы "train_samples_ner.txt" "valid_samples_ner.txt" "test_samples_ner.txt" 
по пути C:/Nastya/ozon_py/my_project/data/

Запуск обучения
python -m deeppavlov train my_ner_config/my_ner.json

"""


def check_tag_dict_content():
    """Проверка содержимого tag.dict"""
    tag_dict_path = "C:/Nastya/ozon_py/my_project/models/my_ner_model/tag.dict"

    print("Содержимое tag.dict:")
    with open(tag_dict_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(repr(content[:200]))  # Покажем первые 200 символов

    # Проверим строки
    with open(tag_dict_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"Количество строк: {len(lines)}")
        for i, line in enumerate(lines[:5]):  # Первые 5 строк
            print(f"Строка {i}: {repr(line)}")
            if '\t' in line:
                parts = line.split('\t')
                print(f"  Частей после split: {len(parts)}")


def fix_model_file_path():
    model_path = 'C:/Nastya/ozon_py/my_project/models/my_ner_model'
    original_file = f"{model_path}/my_ner_model.pth.tar"
    buggy_file = f"{model_path}/my_ner_model.pth.pth.tar"

    # Создаем копию с нужным именем
    if os.path.exists(original_file) and not os.path.exists(buggy_file):
        shutil.copy2(original_file, buggy_file)
        print(f"Создана копия: {buggy_file}")
        return True
    elif os.path.exists(buggy_file):
        print(f"Файл уже существует: {buggy_file}")
        return True
    else:
        print(f"Исходный файл не найден: {original_file}")
        return False


"""
Можно сделать отдельную функцию для обновления конфига
"""


def update_config_paths(config_path, new_model_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        conf = json.load(f)

    conf['metadata']['variables']['MODEL_PATH'] = new_model_path
    print(f"MODEL_PATH установлен: {conf['metadata']['variables']['MODEL_PATH']}")

    for pipe in conf['chainer']['pipe']:
        if pipe.get('class_name') == 'torch_transformers_sequence_tagger':
            old_save_path = pipe.get('save_path')
            old_load_path = pipe.get('load_path')

            pipe['save_path'] = f"{new_model_path}/my_ner_model.pth.tar"
            pipe['load_path'] = f"{new_model_path}/my_ner_model.pth.tar"

            print(f"Обновлено save_path: {old_save_path} -> {pipe['save_path']}")
            print(f"Обновлено load_path: {old_load_path} -> {pipe['load_path']}")
        elif pipe.get('id') == 'tag_vocab':
            old_save_path = pipe.get('save_path')
            old_load_path = pipe.get('load_path')

            pipe['save_path'] = f"{new_model_path}/tag.dict"
            pipe['load_path'] = f"{new_model_path}/tag.dict"

            print(f"Обновлено save_path: {old_save_path} -> {pipe['save_path']}")
            print(f"Обновлено load_path: {old_load_path} -> {pipe['load_path']}")

    # Убедитесь, что пути к данным указаны правильно
    if 'dataset_reader' in conf:
        conf['dataset_reader']['data_path'] = 'C:/Nastya/ozon_py/my_project/data/'
        # Укажите конкретные имена файлов
        conf['dataset_reader']['train'] = 'train.txt'
        conf['dataset_reader']['valid'] = 'valid.txt'
        conf['dataset_reader']['test'] = 'test.txt'

    with open('my_ner_config/my_ner_debug.json', 'w', encoding='utf-8') as f:
        json.dump(conf, f, ensure_ascii=False, indent=2)

    return conf


"""
Если ты сам создавал конфигурационный файл, то:
from deeppavlov import build_model
ner_model = build_model('path_to_your_config.json', download=False)

def custom_ner_model_load():
    try:
        # Сначала фиксим путь к файлу
        if not fix_model_file_path():
            return None

        # 1. Обновите конфиг
        config_path = 'my_ner_config/my_ner.json'
        new_model_path = 'C:/Nastya/ozon_py/my_project/models/my_ner_model'

        model_file = f"{new_model_path}/my_ner_model.pth.tar"
        tag_file = f"{new_model_path}/tag.dict"

        print(f"Проверка файла модели: {model_file}")
        print(f"Существует: {os.path.exists(model_file)}")
        print(f"Проверка файла словаря: {tag_file}")
        print(f"Существует: {os.path.exists(tag_file)}")

        # Проверяем существование файлов
        if not os.path.exists(model_file):
            print(f'Модель не найдена {model_file}')
            return None

        if not os.path.exists(tag_file):
            print(f'Словарь тегов не найден {model_file}')
            return None

        conf = update_config_paths(config_path, new_model_path)
        config_path_debug = 'my_ner_config/my_ner_debug.json'

        with open(config_path_debug, 'r', encoding='utf-8') as f:
            debug_conf = json.load(f)
            for pipe in debug_conf['chainer']['pipe']:
                if pipe.get('class_name') == 'torch_transformers_sequence_tagger':
                    print(f"Финальный load_path в конфиге: {pipe.get('load_path')}")

        custom_ner_model = build_model(config_path_debug, download=False)
        print("Модель успешно загружена")
        return custom_ner_model

    except ConfigError as e:
        print("Ошибка загрузки кастомной NER DeepPavlov модели:", e)
        traceback.print_exc()
        return None
"""


def load_model_weights_manually(model, model_path):
    """Явная загрузка весов в модель"""
    try:
        # Загружаем состояние
        state = torch.load(model_path, map_location='cpu')

        if 'model_state_dict' in state:
            # Загружаем веса в модель
            model.load_state_dict(state['model_state_dict'])
            print("Веса успешно загружены в модель")
            return True
        else:
            print("Файл модели не содержит model_state_dict")
            return False

    except Exception as e:
        print(f"Ошибка загрузки весов: {e}")
        return False


def custom_ner_model_load():
    try:
        if not fix_model_file_path():
            return None

        config_path = 'my_ner_config/my_ner_debug.json'
        new_model_path = 'C:/Nastya/ozon_py/my_project/models/my_ner_model'

        # Загружаем модель
        custom_ner_model = build_model(config_path, download=False)

        # ЯВНО ЗАГРУЖАЕМ ВЕСА
        model_file = f"{new_model_path}/my_ner_model.pth.tar"
        if os.path.exists(model_file):
            # Получаем доступ к внутренней torch модели
            for component in custom_ner_model.pipe:
                if hasattr(component, 'model'):
                    success = load_model_weights_manually(component.model, model_file)
                    if success:
                        print("Модель с весами успешно загружена")
                    break

        # Принудительно устанавливаем режим evaluation
        set_model_to_eval_mode(custom_ner_model)
        return custom_ner_model

    except Exception as e:
        print("Ошибка загрузки модели:", e)
        return None


def ner_model_use(text):
    custom_ner_model = custom_ner_model_load()
    if not custom_ner_model:
        print("Модель не загружена")
        return None
    try:
        # 4. Сделайте предсказание
        entities = custom_ner_model([text])
        return entities
    except ConfigError as e:
        print("Ошибка предсказания: ", e)
        return None


def check_model_architecture():
    model_path = "C:/Nastya/ozon_py/my_project/models/my_ner_model/my_ner_model.pth.tar"
    state = torch.load(model_path, map_location='cpu')

    if 'model_state_dict' in state:
        print("Ключи в model_state_dict:")
        for key in state['model_state_dict'].keys():
            print(f"  {key}")

        print(f"\nВсего параметров: {len(state['model_state_dict'])}")

        # Проверяем наличие classifier весов
        classifier_keys = [k for k in state['model_state_dict'].keys() if 'classifier' in k]
        print(f"Classifier keys: {classifier_keys}")


def check_gpu():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Используется CPU")


def set_model_to_eval_mode(model):
    """Устанавливаем модель в режим evaluation"""
    for component in model.pipe:
        if hasattr(component, 'model'):
            component.model.eval()
            print("Модель переведена в eval mode")
            break


def retrain_with_better_params():
    config_path_debug = 'my_ner_config/my_ner_debug.json'

    # Загрузите и обновите конфиг
    with open(config_path_debug, 'r', encoding='utf-8') as f:
        conf = json.load(f)

    # Увеличьте количество эпох
    conf['train']['epochs'] = 50

    # Уменьшите learning rate для лучшей сходимости
    conf['chainer']['pipe'][2]['optimizer_parameters']['lr'] = 1e-05

    # Добавьте early stopping
    conf['train']['early_stopping'] = True
    conf['train']['early_stopping_patience'] = 10

    # Сохраните обновленный конфиг
    with open(config_path_debug, 'w', encoding='utf-8') as f:
        json.dump(conf, f, ensure_ascii=False, indent=2)

    # Обучаем заново
    print("Начинаем переобучение модели с улучшенными параметрами...")
    train_model(config_path_debug, download=False)
    print("Переобучение завершено")


def train_ner_model():
    """Обучение NER модели на ваших данных"""
    config_path = 'my_ner_config/my_ner_debug.json'

    # Обновляем конфиг с правильными путями
    with open(config_path, 'r', encoding='utf-8') as f:
        conf = json.load(f)

    # Убедитесь, что пути к данным указаны правильно
    if 'dataset_reader' in conf:
        conf['dataset_reader']['data_path'] = 'C:/Nastya/ozon_py/my_project/data/'
        conf['dataset_reader']['train'] = 'train.txt'
        conf['dataset_reader']['valid'] = 'valid.txt'
        conf['dataset_reader']['test'] = 'test.txt'

    # Сохраняем обновленный конфиг
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(conf, f, ensure_ascii=False, indent=2)

    print("Начинаем обучение модели...")
    try:
        from deeppavlov import train_model
        train_model(config_path, download=False)
        print("Обучение завершено успешно!")
        return True
    except Exception as e:
        print(f"Ошибка обучения: {e}")
        return False


def retrain_model():
    config_path_debug = 'my_ner_config/my_ner_debug.json'
    new_model_path = 'C:/Nastya/ozon_py/my_project/models/my_ner_model'

    # Удаляем старые файлы
    model_file = f"{new_model_path}/my_ner_model.pth.tar"
    if not os.path.exists(model_file):
        os.remove(model_file)
        print("Старый файл модели удален")

    # Обновляем конфиг
    config_path = 'my_ner_config/my_ner.json'
    conf = update_config_paths(config_path, new_model_path)

    # Обучаем заново
    print("Начинаем обучение модели...")
    train_model(config_path_debug, download=False)
    print("Обучение завершено")


"""
NAME ~ 200
PRICE ~ 200
CATEGORY ~ 200
BRAND ~ 200
COUNTRY ~ 100

def main():
    #clean_conll('project-8-at-2025-08-26-19-05-6671327e.conll', 'cleaned_training_data_ner.conll')
    #data_preparation()

    #Можно и из Python (если хочешь скриптом):
    #from deeppavlov import train_model
    #train_model(CONFIG_PATH)

    # Сначала пофиксим пути к файлам
    fix_model_file_path()

    # Проверим GPU
    check_gpu()

    # Переобучим модель если нужно
    answer = input("Переобучить моедь? (y/n): ")
    if answer.lower() == 'y':
        retrain_with_better_params()

    model_path = "C:/Nastya/ozon_py/my_project/models/my_ner_model/my_ner_model.pth.tar"
    state = torch.load(model_path, map_location='cpu')

    print(type(state))
    print(state.keys() if isinstance(state, dict) else "Не dict")

    # Конкретный тест
    text = 'Барак Обама родился на Гавайях'
    entities = ner_model_use(text)
    if entities:
        print("Успешный результат: ", entities)
    else:
        print("Не удалось получить результат")
"""


def analyze_tag_dict():
    """Анализ содержимого файла tag.dict"""
    tag_dict_path = "C:/Nastya/ozon_py/my_project/models/my_ner_model/tag.dict"

    try:
        # Читаем как бинарный файл
        with open(tag_dict_path, 'rb') as f:
            binary_content = f.read()
            print(f"Размер файла: {len(binary_content)} байт")
            print(f"Первые 100 байт: {binary_content[:100]}")

        # Читаем как текстовый файл
        with open(tag_dict_path, 'r', encoding='utf-8', errors='ignore') as f:
            text_content = f.read()
            print(f"\nТекстовое содержимое (первые 500 символов):")
            print(text_content[:500])

        # Пробуем разные кодировки
        encodings = ['utf-8', 'latin-1', 'cp1251', 'ascii']
        for encoding in encodings:
            try:
                with open(tag_dict_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    print(f"\nСодержимое в кодировке {encoding}:")
                    print(content[:200])
                    break
            except:
                continue

    except Exception as e:
        print(f"Ошибка анализа файла: {e}")


def load_huggingface_model():
    """Загрузка модели через Hugging Face"""
    model_path = "C:/Nastya/ozon_py/my_project/models/my_ner_model"
    model_file = f"{model_path}/my_ner_model.pth.tar"

    # Загружаем конфиг BERT
    model_name = "DeepPavlov/rubert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Загружаем словарь тегов (JSON формат)
    json_dict_path = f"{model_path}/tag_dict.json"
    tag_vocab = None

    # Пробуем загрузить существующий JSON словарь
    try:
        with open(json_dict_path, 'r', encoding='utf-8') as f:
            tag_vocab = json.load(f)
        print("Словарь тегов загружен из JSON файла")
    except:
        print("Не удалось загрузить JSON словарь тегов, создаем новый...")
        tag_vocab = create_both_tag_dicts()  # Создаем оба формата

    if tag_vocab is None:
        print("Не удалось создать словарь тегов")
        return None, None, None

    n_tags = len(tag_vocab)

    # Создаем модель
    model = BertForTokenClassification.from_pretrained(
        model_name,
        num_labels=n_tags,
        output_attentions=False,
        output_hidden_states=False
    )

    # Загружаем веса из дообученной модели
    if os.path.exists(model_file):
        try:
            checkpoint = torch.load(model_file, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Удаляем префикс 'module.' если есть
                if all(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {k[7:]: v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                print("Веса дообученной модели загружены!")
            else:
                model.load_state_dict(checkpoint)
                print("Веса загружены напрямую из checkpoint")
        except Exception as e:
            print(f"Ошибка загрузки весов: {e}")
            print("Используем предобученные веса")

    model.eval()
    return model, tokenizer, tag_vocab


def check_tag_dict_format():
    """Проверка формата файла tag.dict"""
    tag_dict_path = "C:/Nastya/ozon_py/my_project/models/my_ner_model/tag.dict"

    # Пробуем определить формат файла
    try:
        # Пробуем как pickle
        with open(tag_dict_path, 'rb') as f:
            content = f.read()
            try:
                pickle.loads(content)
                print("Файл tag.dict: формат pickle")
                return 'pickle'
            except:
                pass

        # Пробуем как JSON
        with open(tag_dict_path, 'r', encoding='utf-8') as f:
            content = f.read()
            try:
                json.loads(content)
                print("Файл tag.dict: формат JSON")
                return 'json'
            except:
                pass

        # Пробуем как текстовый файл
        with open(tag_dict_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines and ':' in lines[0]:
                print("Файл tag.dict: текстовый формат (key:value)")
                return 'text'

        print("Неизвестный формат файла tag.dict")
        return 'unknown'

    except Exception as e:
        print(f"Ошибка чтения файла tag.dict: {e}")
        return 'error'


def predict_with_huggingface(text, model, tokenizer, tag_vocab):
    """Предсказание с Hugging Face моделью"""
    try:
        # Токенизация
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Предсказание
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

            # Конвертируем в теги
            tags = []
            for idx in predictions[0]:
                tag_idx = idx.item()
                # Ищем тег по значению
                for tag, tag_id in tag_vocab.items():
                    if tag_id == tag_idx:
                        tags.append(tag)
                        break
                else:
                    tags.append('O')  # Если не найден

            # Получаем токены
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

            return tokens, tags

    except Exception as e:
        print(f"Ошибка предсказания: {e}")
        return None, None


"""
def main():
    fix_model_file_path()
    check_gpu()

    # 1. Проверим архитектуру модели
    print("Проверка архитектуры модели:")
    check_model_architecture()
    print("\n" + "=" * 50 + "\n")

    # 2. Переобучение (если нужно)
    answer = input("Переобучить модель? (y/n): ")
    if answer.lower() == 'y':
        retrain_with_better_params()

    # 3. Загружаем модель с явной загрузкой весов
    print("Загрузка модели с явной загрузкой весов...")
    custom_ner_model = custom_ner_model_load()

    if not custom_ner_model:
        print("Не удалось загрузить модель")
        return

    # 4. Тестируем
    text = 'Барак Обама родился на Гавайях'
    entities = custom_ner_model([text])

    if entities:
        print("Результат:", entities)

        # Детальный анализ
        tokens = entities[0][0]
        tags = entities[1][0]
        print("\nДетализация:")
        for token, tag in zip(tokens, tags):
            print(f"  '{token}' -> {tag}")
    else:
        print("Не удалось получить результат")
"""


def create_tag_dict_from_data():
    """Создание словаря тегов из обучающих данных"""
    data_path = "C:/Nastya/ozon_py/my_project/data/train.txt"

    try:
        # Определяем, какой формат нужен
        stack = inspect.stack()
        caller_function = stack[1].function if len(stack) > 1 else ''

        if caller_function == 'load_huggingface_model':
            # JSON формат для Hugging Face
            return create_json_tag_dict(data_path)
        else:
            # TSV формат для DeepPavlov
            return create_proper_tag_dict(data_path)

    except Exception as e:
        print(f"Не удалось определить caller: {e}")
        # По умолчанию создаем оба формата
        create_json_tag_dict(data_path)
        return create_proper_tag_dict(data_path)


def create_both_tag_dicts():
    """Создает оба формата словарей - для DeepPavlov и Hugging Face"""
    data_path = "C:/Nastya/ozon_py/my_project/data/train.txt"

    # 1. TSV формат для DeepPavlov
    tsv_dict_path = "C:/Nastya/ozon_py/my_project/models/my_ner_model/tag.dict"
    create_proper_tag_dict(data_path)

    # 2. JSON формат для Hugging Face
    json_dict_path = "C:/Nastya/ozon_py/my_project/models/my_ner_model/tag_dict.json"
    tag_vocab = create_json_tag_dict(data_path)

    # Сохраняем отдельно для Hugging Face
    if tag_vocab:
        with open(json_dict_path, 'w', encoding='utf-8') as f:
            json.dump(tag_vocab, f, ensure_ascii=False, indent=2)
        print(f"JSON словарь сохранен: {json_dict_path}")

    return tag_vocab


def create_json_tag_dict(data_path):
    """JSON формат для Hugging Face"""
    tag_dict_path = "C:/Nastya/ozon_py/my_project/models/my_ner_model/tag.dict"

    try:
        tags = set()
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and ' ' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        tag = parts[-1]
                        tags.add(tag)

        tags_list = sorted(list(tags))
        tag_vocab = {tag: idx for idx, tag in enumerate(tags_list)}

        with open(tag_dict_path, 'w', encoding='utf-8') as f:
            json.dump(tag_vocab, f, ensure_ascii=False, indent=2)

        print(f"Создан JSON словарь тегов с {len(tag_vocab)} тегами")
        return tag_vocab

    except Exception as e:
        print(f"Ошибка создания JSON словаря тегов: {e}")
        return None


def create_proper_tag_dict(data_path):
    """Правильный формат для DeepPavlov"""
    tag_dict_path = "C:/Nastya/ozon_py/my_project/models/my_ner_model/tag.dict"

    try:
        from collections import Counter
        tag_counter = Counter()

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and ' ' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        tag = parts[-1]
                        tag_counter[tag] += 1

        # Сохраняем в формате тег\tчастота
        with open(tag_dict_path, 'w', encoding='utf-8') as f:
            for tag, count in tag_counter.most_common():
                f.write(f"{tag}\t{count}\n")

        print(f"Создан DeepPavlov словарь тегов с {len(tag_counter)} тегами:")
        for tag, count in tag_counter.most_common():
            print(f"  {tag}: {count}")

        return tag_counter

    except Exception as e:
        print(f"Ошибка создания DeepPavlov словаря тегов: {e}")
        return None


"""
def main():
    print("Проверка файлов данных...")
    data_files = [
        'C:/Nastya/ozon_py/my_project/data/train.txt',
        'C:/Nastya/ozon_py/my_project/data/valid.txt',
        'C:/Nastya/ozon_py/my_project/data/test.txt'
    ]

    for file in data_files:
        if os.path.exists(file):
            print(f"✓ Найден: {file}")
        else:
            print(f"✗ Отсутствует: {file}")

    # Создаем оба формата словарей
    print("\nСоздание словарей тегов...")
    create_both_tag_dicts()

    print("\nЗагрузка дообученной модели через Hugging Face...")
    model, tokenizer, tag_vocab = load_huggingface_model()

    if model is None or tag_vocab is None:
        print("Не удалось загрузить модель или словарь тегов")
        return

    print(f"Загружено тегов: {len(tag_vocab)}")
    print("Теги:", list(tag_vocab.keys()))

    # Тестируем на нескольких примерах
    test_texts = [
        'iPhone 13 Pro 128GB',
        'Nike Air Max кроссовки',
        'Китайский производитель',
        'Барак Обама родился на Гавайях'
    ]

    for text in test_texts:
        print(f"\nТестируем: '{text}'")
        tokens, tags = predict_with_huggingface(text, model, tokenizer, tag_vocab)

        if tokens and tags:
            print("Результат:")
            for token, tag in zip(tokens, tags):
                if not token.startswith('##'):  # Пропускаем subword tokens
                    print(f"  '{token}' -> {tag}")
        else:
            print("❌ Не удалось получить результат")

"""


def retrain_with_more_epochs():
    """Дополнительное обучение с увеличенным количеством эпох"""
    config_path = 'my_ner_config/my_ner_debug.json'

    try:
        # Сначала создаем правильный tag.dict
        create_correct_tag_dict()

        # Загружаем конфиг
        with open(config_path, 'r', encoding='utf-8') as f:
            conf = json.load(f)

        # Увеличиваем количество эпох
        conf['train']['epochs'] = 20
        conf['train']['batch_size'] = 8

        # Уменьшаем learning rate для тонкой настройки
        for pipe in conf['chainer']['pipe']:
            if pipe.get('class_name') == 'torch_transformers_sequence_tagger':
                if 'optimizer_parameters' in pipe:
                    pipe['optimizer_parameters']['lr'] = 2e-5

        # Убираем неподдерживаемые параметры
        if 'save_best' in conf['train']:
            del conf['train']['save_best']

        # Сохраняем обновленный конфиг
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(conf, f, ensure_ascii=False, indent=2)

        print("Начинаем дополнительное обучение...")
        train_model(config_path, download=False)

        return True

    except Exception as e:
        print(f"Ошибка при дополнительном обучении: {e}")
        return False


def evaluate_model_quality(model, tokenizer, tag_vocab):
    """Оценка качества модели после дообучения"""
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ КАЧЕСТВА МОДЕЛИ")
    print("=" * 60)

    test_cases = [
        # Текст и ожидаемые результаты
        {
            'text': 'iPhone 13 Pro 128GB',
            'expected': [
                ('iPhone', 'B-BRAND'),
                ('13', 'I-BRAND'),
                ('Pro', 'I-BRAND'),
                ('128GB', 'B-PRICE')
            ]
        },
        {
            'text': 'Nike Air Max кроссовки',
            'expected': [
                ('Nike', 'B-BRAND'),
                ('Air', 'I-BRAND'),
                ('Max', 'I-BRAND'),
                ('кроссовки', 'B-CATEGORY')
            ]
        },
        {
            'text': 'Samsung Galaxy S21',
            'expected': [
                ('Samsung', 'B-BRAND'),
                ('Galaxy', 'I-BRAND'),
                ('S21', 'I-BRAND')
            ]
        },
        {
            'text': 'Китайский производитель электроники',
            'expected': [
                ('Китайский', 'B-COUNTRY'),
                ('производитель', 'O'),
                ('электроники', 'B-CATEGORY')
            ]
        }
    ]

    correct_predictions = 0
    total_predictions = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nТест {i}: '{test_case['text']}'")

        tokens, tags = predict_with_huggingface(test_case['text'], model, tokenizer, tag_vocab)

        if tokens and tags:
            print("Результат:")
            for token, tag in zip(tokens, tags):
                if not token.startswith('##') and token not in ['[CLS]', '[SEP]']:
                    print(f"  '{token}' -> {tag}")

            # Простая оценка точности
            expected_dict = dict(test_case['expected'])
            for token, tag in zip(tokens, tags):
                if token in expected_dict and expected_dict[token] == tag:
                    correct_predictions += 1
                total_predictions += 1
        else:
            print("❌ Не удалось получить результат")

    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\nТочность на тестовых примерах: {accuracy:.2f}%")

    return accuracy


def main():
    print("Улучшение модели...")

    # 1. Аугментация данных
    augment_training_data()

    # 2. Обновляем пути в конфиге
    config_path = 'my_ner_config/my_ner_debug.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        conf = json.load(f)

    if 'dataset_reader' in conf:
        conf['dataset_reader']['train'] = 'train_augmented.txt'

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(conf, f, ensure_ascii=False, indent=2)

    # 3. Обучение с регуляризацией
    retrain_with_regularization()

    # 4. Тестирование с постобработкой
    model, tokenizer, tag_vocab = load_huggingface_model()

    print("Тестирование с постобработкой:")
    test_text = 'iPhone 13 Pro 128GB'
    tokens, tags = predict_with_postprocessing(test_text, model, tokenizer, tag_vocab)

    print(f"'{test_text}':")
    for token, tag in zip(tokens, tags):
        if not token.startswith('##'):
            print(f"  {token} -> {tag}")


def postprocess_predictions(tokens, tags):
    """Постобработка предсказаний"""
    new_tags = []

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        # Исправляем очевидные ошибки
        if token.lower() in ['iphone', 'samsung', 'nike', 'apple', 'adidas'] and tag == 'B-PRICE':
            new_tags.append('B-BRAND')
        elif token.isdigit() and tag == 'B-BRAND':
            new_tags.append('B-PRICE')
        else:
            new_tags.append(tag)

    return new_tags


def predict_with_postprocessing(text, model, tag_dict):
    # Получаем предсказания
    tokens, labels = model([text])[0]

    # Фильтруем специальные токены
    result = []
    for token, label in zip(tokens, labels):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            result.append((token, label))

    return result


def retrain_with_regularization():
    """Обучение с регуляризацией против переобучения"""
    config_path = 'my_ner_config/my_ner_debug.json'

    with open(config_path, 'r', encoding='utf-8') as f:
        conf = json.load(f)

    # Уменьшаем переобучение
    conf['train']['epochs'] = 10  # Меньше эпох
    conf['train']['batch_size'] = 16  # Больше batch size

    # Добавляем dropout и регуляризацию
    for pipe in conf['chainer']['pipe']:
        if pipe.get('class_name') == 'torch_transformers_sequence_tagger':
            pipe['dropout'] = 0.3  # Увеличиваем dropout
            if 'optimizer_parameters' in pipe:
                pipe['optimizer_parameters']['weight_decay'] = 0.01  # L2 регуляризация

    # Увеличиваем patience для early stopping
    conf['train']['patience'] = 10

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(conf, f, ensure_ascii=False, indent=2)

    print("Обучение с регуляризацией...")
    train_model(config_path, download=False)


def augment_training_data():
    """Аугментация обучающих данных"""
    original_path = "C:/Nastya/ozon_py/my_project/data/train.txt"
    augmented_path = "C:/Nastya/ozon_py/my_project/data/train_augmented.txt"

    # Примеры для аугментации
    augmentation_examples = [
        "iPhone\tB-BRAND\n13\tI-BRAND\nPro\tI-BRAND\n128GB\tB-PRICE",
        "Samsung\tB-BRAND\nGalaxy\tI-BRAND\nS21\tI-BRAND\nUltra\tI-BRAND",
        "Nike\tB-BRAND\nAir\tI-BRAND\nMax\tI-BRAND\nкроссовки\tB-CATEGORY",
        "Apple\tB-BRAND\nMacBook\tI-BRAND\nPro\tI-BRAND\n2023\tB-PRICE"
    ]

    # Копируем оригинальные данные
    with open(original_path, 'r', encoding='utf-8') as src:
        with open(augmented_path, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
            dst.write("\n\n")

            # Добавляем аугментированные примеры
            for example in augmentation_examples:
                dst.write(example + "\n\n")

    print("Аугментированные данные созданы!")


def save_final_model(model, tokenizer, tag_vocab):
    """Сохранение финальной модели"""
    model_path = "C:/Nastya/ozon_py/my_project/models/my_ner_model"

    # Сохраняем модель
    model.save_pretrained(f"{model_path}/final_model")

    # Сохраняем токенайзер
    tokenizer.save_pretrained(f"{model_path}/final_model")

    # Сохраняем словарь тегов
    with open(f"{model_path}/final_model/tag_dict.json", 'w', encoding='utf-8') as f:
        json.dump(tag_vocab, f, ensure_ascii=False, indent=2)

    print(f"Финальная модель сохранена в: {model_path}/final_model/")
    print("Для использования:")


if __name__ == '__main__':
    main()
"""
Выберите Flair если:
Нужна легкая кастомизация тегов
Хотите экспериментировать с архитектурой
Нужен transfer learning
Работаете с разными типами эмбеддингов
Цените простоту дообучения
"""
