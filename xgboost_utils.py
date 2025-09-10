import pickle
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

def train_XGBoost(sample_df):
    """
    Мультикатегорийность и семантика
    В реальности товары могут относиться к нескольким категориям — попробуй модель мультиклассификации
    на названии, чтобы предложить правильные категории, а потом сравнивать с текущей категорией.

    Раз у тебя уже есть вектора name_vec и category_vec, то можно реализовать модель для предсказания категории по
    названию (мультиклассификация) прямо на этих векторах, без текстов.
    """
    counts = sample_df['category'].value_counts()
    valid_categories = counts[counts >= 5].index
    filtered_df = sample_df[sample_df['category'].isin(valid_categories)]

    X = np.vstack(filtered_df['name_vec'].values)
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
        eval_metric='mlogloss',
        seed=42
    )

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    true_labels = unique_labels(Y_test, Y_pred)
    true_class_names = label_encoder.inverse_transform(true_labels)
    print(classification_report(Y_test, Y_pred, labels=true_labels, target_names=true_class_names, zero_division=0))

    return model, label_encoder

'''
train_save_XGBoost
Запись в бинарный файл обученного XGBoost
'''

def save_XGBoost(model, label_encoder, xgboost_path = 'xgb_classifier.pkl', label_encoder_path = 'label_encoder.pkl'):

    with open(xgboost_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Сохранено: {xgboost_path}")

    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

    print(f"Сохранено: {label_encoder_path}")

'''
load_Tfidf
Загрузка обученного XGBoost
'''

def load_XGBoost():

    with open('xgb_classifier.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    return model, label_encoder