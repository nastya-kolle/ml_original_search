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
#df_copy.duplicated(subset='id').sum()

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
len(df_copy[df_copy.rating>percentile_90_value])

sns.histplot(rating[rating>percentile_90_value], bins=20, color='blue')
plt.title('Гистограмма распределения рейтинга товаров')
plt.xlabel('Рейтинг')
plt.ylabel('Количество')
plt.show()

plt.boxplot(rating[rating>percentile_90_value], vert=False)
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

df_copy['category_vec'] = df_copy['category'].apply(model.get_sentence_vector)
df_copy['name_vec'] = df_copy['name'].apply(model.get_sentence_vector)

def cosine_sim(v1, v2):
  return cosine_similarity([v1], [v2])[0][0]

df_copy['cos_sim'] = df_copy.apply(lambda row: cosine_sim(row['category_vec'], row['name_vec']))

"""Значение от -1 до 1.

Чем ниже, тем меньше заголовок похож на категорию.

Можно использовать как признак: cos_sim
"""

