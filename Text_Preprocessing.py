import re
import ijson
import requests
import num2words
import fasttext
import numpy as np
import pandas as pd
import natasha
import json
# nltk.download('stopwords')
# import nltk for stopwords
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from bs4 import BeautifulSoup
from num2words import num2words
from razdel import tokenize
from spellchecker import SpellChecker
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, Doc, MorphVocab, NewsNERTagger
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from tfidf_utils import load_Tfidf
from model_loader import model


class Text_Preprocessing():

    def __init__(self):
        self.morph = MorphAnalyzer()
        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.news_tagger = NewsMorphTagger(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)
        self.morph_vocab = MorphVocab()

        self.stop_words = self.StopWords(self.morph)

    '''
    Normalization
    1) Приведение к нижнему регистру 
    НО: может потретяться важность имени брендов (Nike -> nike)
    2) Удалять !, ?, ...
    НО: сохранить %, $
    3) Удалить все точки, кроме точек в дробных числах
    4) Удалить email
    5) Удалить html теги
    6) Удаляем пробелы
    7) Цифры преобразовываем в слова, т к они важны
    8) Удалить стоп-слова
    9) Удалить неинформатиивные сокращения ( и т д , и т п)
    10) Расшифровать сокращения, несущие некоторый смысл
    11) Исправить явные технические опечатки (адресс - адрес)
    '''

    def Normalization(self, string, stop_words):

        # convert to lower case
        lower_string = string.lower()

        # замена нужных сокращений
        '''
        to_replace = {
            "т.е.": "то есть",
            "т.к.": "так как",
            "т.о.": "таким образом"
        }
        for word, replacement in to_replace.items():
            if word in lower_string:
                print(f"Заменяю '{word}' на '{replacement}'")
                lower_string = lower_string.replace(word, replacement)
        '''
        # удаление неинформативных сокращений
        to_delete = ["и т.д.", "и т.п.", "и др.", "т.о.", "т.н.", "см. также", "т.е.", "т.к."]
        for word in to_delete:
            if word in lower_string:
                lower_string = lower_string.replace(word, ' ')

        # исправление явных опечаток
        '''
        if morph is None:
            morph = MorphAnalyzer()

        if spell is None:
            spell = SpellChecker(language ="ru")
        tokens = list(tokenize(lower_string))
        new_text=list()
        for token in tokens:
            word = token.text.lower()
            if not morph.word_is_known(word):
                word_text = spell.correction(word)
                new_text.append(word_text if word_text else word)
            else:
                new_text.append(word)
        reconstructed_str = ' '.join(new_text)
        '''
        # удаление email
        cleaned_email = re.sub(r"\S+@\S+", "", lower_string)

        # удаление html тегов
        soup = BeautifulSoup(cleaned_email, "html.parser")
        cleaned_html = soup.get_text()

        # удаление точек, которые разделяют предложения
        cleaned_dots = re.sub(r"(?<!\d)\.(?!\d)", "", cleaned_html)

        # Оставляем: буквы, цифры, пробелы, денежные знаки (% $ € £ ¥ ₽ и т.п.) и точки(дробные числа)
        cleaned_text = re.sub(r"[^\w\s.%$€£¥₽]", "", cleaned_dots)

        # remove white spaces
        no_wspace_cleaned_text = cleaned_text.strip()

        # Если это цифра,то ее необходимо преобразовать в слово
        '''
        tokens = list(tokenize(no_wspace_cleaned_text))
        tokens_with_numbers = list()
        for token in tokens:
            word = token.text.lower()
            word_clean = word.replace(",",".")
            try:
                if "." in word_clean:
                    number = float(word_clean)
                    integer_number = int(number)
                    fractional_number = word_clean.split(".")[1]
                    fractional_number = int(fractional_number)

                    word_integer_number = num2words(integer_number, lang='ru')
                    word_fractional_number = num2words(int(fractional_number), lang='ru')
                    combined_number= f"{word_integer_number} целых {word_fractional_number} дробных"
                    tokens_with_numbers.append(combined_number)

                else:
                    integer_number = int(word_clean)
                    word_text = num2words(integer_number, lang='ru')  # для русского
                    tokens_with_numbers.append(word_text)
            except ValueError:
                tokens_with_numbers.append(word)


        # Просто соединяем токены через пробел
        reconstructed = ' '.join(tokens_with_numbers)

        # convert string to list of words
        lst_string = [reconstructed][0].split()
        '''
        # Если не нужно преобразовывать числа в слова, просто продолжаем с текущим текстом
        reconstructed = no_wspace_cleaned_text

        # convert string to list of words
        lst_string = [reconstructed][0].split()

        # remove stopwords
        # stop_words = StopWords()
        no_stpwords_string = ""

        for word in reconstructed.split():
            if word not in stop_words:
                no_stpwords_string += word + ' '

        # removing last space
        no_stpwords_string = no_stpwords_string.strip()

        # output
        return no_stpwords_string

    '''
    StopWords
    1) Загрузка стоп-слов с GitHub
    2) Инициализация морфоанализатора
    3) Фильтрация числительных и несловесного мусора
    4) сохранить в файл
    '''

    def StopWords(self, morph):
        # Загрузка стоп-слов с GitHub
        '''
        Стоп слова с githab
        url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ru/master/stopwords-ru.txt"
        '''
        with open("stopwords_ru.txt", encoding="utf-8") as f:
            stopwords_list = f.read().splitlines()

        # Фильтрация числительных и несловесного мусора
        filtered_stopwords = set()
        for word in stopwords_list:
            if not word.isalpha():  # Убираем слова с цифрами, пунктуацией и т.п.
                continue

            parsed_word = morph.parse(word)[0]
            if parsed_word.tag.POS == 'NUMR':  # Убираем числительные (NUMR)
                continue

            filtered_stopwords.add(parsed_word.normal_form)

        # сохранить в файл
        with open("stopwords_ru_without_numbers.txt", "w", encoding="utf-8") as f:
            for word in sorted(filtered_stopwords):
                f.write(word + "\n")

        return filtered_stopwords

    '''
    Tokenization_Lemmatization
    Natasha
    1) Токенизация 
    2) Лемматизация
    '''

    def Tokenization_Lemmatization(self, string):
        if not string.strip():  # защита от пустых строк
            return []

        doc = Doc(string)

        # токенизация
        doc.segment(self.segmenter)

        # Морфологический разбор
        doc.tag_morph(self.news_tagger)

        # лемматизация
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)

        tokens = [token.lemma if token.lemma else token.text for token in doc.tokens if
                  token.pos not in {'PUNCT', 'SPACE'}]

        return tokens

    '''
    Named_Entity_Recognition
    Natasha с дообучением 

    Без дообучения определяет сущности:
    PER — Person (персона, человек)
    LOC — Location (место, страна, город и т.д.)
    ORG — Organization (организации)
    DATE — Даты

    Должны быть сущности:
    1) PRD - Product (Наименование продукта)
    2) BR - Brand (брэнд, производитель)
    3) CAT - Category (Категория товара)
    4) VOL - Volume (Обьем)
    5) S - Size (Размер)
    6) MAT - Material (Материал)
    7) PR - Price (Цена)
    8) CNTR -Country (Страна производства)
    9) CERT -Certification (Лицензии, сертификаты, стандарты)
    '''

    def Named_Entity_Recognition(self, string):

        doc = Doc(string)

        # токенизация
        doc.segment(self.segmenter)

        # NER
        doc.tag_ner(self.ner_tagger)

        entities = []
        for span in doc.spans:
            span.normalize(self.morph_vocab)
            entities.append({
                'text': span.text,
                'type': span.type,
                'normal': span.normal
            })

        return entities

    '''
    Embedding_Mean
    FastText + mean для аггрегации
    '''

    def Embedding_Mean(self, lemmatized_string):

        # каждое слово переводим в вектор (размерность 300)
        vectors = list()
        for token in lemmatized_string:
            vec = model.get_word_vector(token)
            vectors.append(vec)

        # для аггрегации берем среднее из всез векторов
        if vectors:
            embedding = np.mean(vectors, axis=0)
        else:
            embedding = np.zeros(model.get_dimension())
        return embedding

    '''
    Embedding_Tfidf
    FastText + TfIdf для взвешенной аггрегации
    Взвешенная сумма эмбеддингов всех слов, где вес каждого слова — это его значимость по TF-IDF. 
    Часто встречающиеся слова (со слабым весом) вносят меньший вклад, чем важные и редкие термины.
    '''

    def Embedding_Tfidf(self, lemmatized_string):

        # загрузка векторайзера и словаря
        vectorizer, tfidf_dict = load_Tfidf()
        vectors = list()
        weights = list()

        # для аггрегации берем взвешенное среднее из всез векторов с помощью TfIdf
        for token in lemmatized_string:
            weight = tfidf_dict.get(token, 0.0)
            vec = model.get_word_vector(token)
            vectors.append(vec * weight)
            weights.append(weight)
        if vectors:
            embedding = np.sum(vectors, axis=0) / (np.sum(weights) + 1e-8)
        else:
            embedding = np.zeros(model.vector_size)
        return embedding

    '''
    Wildberries Products	Обучение TF-IDF весов, FastText эмбеддинги
    Corpus_Wildberries
    wb-products	Основные поля: название, бренд, описание	Корпус для TF‑IDF
    wb-feedbacks	Основные поля:Отзывы от пользователей о товаре	Дополнительный корпус, особенно для описания подделок
    wb-questions	Основные поля:Вопросы/ответы, связанные с товаром	Доп. контекст и признаки товара
    '''

    def Corpus_Wildberries(self):

        # Загрузка датасета продуктов Wildberries

        records_products = []
        with open("products_new.json", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 50000:
                    break
                records_products.append(json.loads(line))
        df_products = pd.DataFrame(records_products)
        df_products = df_products[['imt_id', 'imt_name', 'subj_name', 'subj_root_name', 'brand_name', 'description']]
        df_products.to_csv("df_products.csv", index=False)

        # Загрузка датасета отзывов Wildberries

        records_feedbacks = []
        with open("feedbacks_new.json", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 50000:
                    break
                records_feedbacks.append(json.loads(line))
        df_feedback = pd.DataFrame(records_feedbacks)
        df_feedback = df_feedback[df_feedback["text"].notna()]
        df_feedback = df_feedback[['nmId', 'productValuation', 'text']]
        df_feedback.to_csv("df_feedback.csv", index=False)

        # Загрузка датасета вопросов и ответов Wildberries

        records_questions = []
        with open("questions_new.json", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 50000:
                    break
                records_questions.append(json.loads(line))
        df_questions = pd.DataFrame(records_questions)
        df_questions = df_questions[df_questions["question"].notna() & df_questions["answer"].notna()]
        df_questions = df_questions[
            ['nmId', 'productName', 'supplierId', 'supplierName', 'brandName', 'question', 'answer']]
        df_questions.to_csv("df_questions.csv", index=False)

        # Агрегация отзывов: собираем все отзывы одного товара в одну строку
        df_feedback_grouped = df_feedback.groupby('nmId')['text'].apply(lambda x: ' '.join(x)).reset_index()

        # Агрегация вопросов + ответов: объединяем их в одну строку на каждый товар
        df_questions['q_and_a'] = df_questions['question'] + ' ' + df_questions['answer']
        df_questions_grouped = df_questions.groupby('nmId')['q_and_a'].apply(lambda x: ' '.join(x)).reset_index()

        df_products.rename(columns={'imt_id': 'nmId'}, inplace=True)
        df_products = df_products.drop_duplicates(subset=['nmId'])

        # Объединение всех данных
        df_products_feedback = df_products.merge(df_feedback_grouped, on='nmId', how='left')
        df_products_feedback_questions = df_products_feedback.merge(df_questions_grouped, on='nmId', how='left')

        # обьединение текста для подачи в корпус
        cols_to_concat = [
            'nmId', 'imt_name', 'subj_name', 'subj_root_name',
            'supplierId', 'supplierName', 'productName',
            'brand_name', 'description', 'text', 'q_and_a', 'productValuation'
        ]
        for col in cols_to_concat:
            if col not in df_products_feedback_questions.columns:
                df_products_feedback_questions[col] = ""

        # Заполним все NaN пустыми строками
        df_products_feedback_questions[cols_to_concat] = df_products_feedback_questions[cols_to_concat].fillna("")

        # Преобразуем всё в строки
        df_products_feedback_questions[cols_to_concat] = df_products_feedback_questions[cols_to_concat].astype(str)

        # Объединяем всё в одну колонку через пробел
        df_products_feedback_questions["text_result"] = df_products_feedback_questions[cols_to_concat].agg(" ".join,
                                                                                                           axis=1)

        df_products_feedback_questions.to_csv("df_products_feedback_questions.csv", index=False)

        corpus = df_products_feedback_questions['text_result'].tolist()
        corpus = corpus[:50000]

        with open("corpus_wildberries.txt", "w", encoding="utf-8") as f:
            for line in corpus:
                f.write(line + "\n")

        # нормализация, токенизация и лемматизация корпуса
        stop_words = self.stop_words
        normalized_corpus = [self.Normalization(text, stop_words) for text in tqdm(corpus)]

        with open("normalized_corpus_wildberries.txt", "w", encoding="utf-8") as f:
            for line in normalized_corpus:
                f.write(line + "\n")

        lemmatized_corpus = [self.Tokenization_Lemmatization(text) for text in tqdm(normalized_corpus)]
        lemmatized_corpus = [" ".join(tokens) for tokens in lemmatized_corpus]

        with open("lemmatized_corpus_wildberries.txt", "w", encoding="utf-8") as f:
            for line in lemmatized_corpus:
                f.write(line + "\n")

        return lemmatized_corpus

    '''
    RuReviews / rureviews	Обогащение словаря подозрительных/типичных слов
    Corpus_RuReviews 
    '''

    def Corpus_RuReviews(self):

        # Загрузка датасета
        df_RuReviews = pd.read_csv('cleaned_kaspi_reviews.csv')

        # небольшая обработка и приведение в нужный вид
        df_RuReviews = df_RuReviews.drop('Unnamed: 0.1', axis=1)
        df_RuReviews.rename(columns={'Unnamed: 0': 'Id'}, inplace=True)
        df_RuReviews = df_RuReviews[df_RuReviews['language'] == 'russian']

        # обьединение текста для подачи в корпус

        cols_to_concat = ['Id', 'language', 'rating', 'category', 'combined_text']
        df_RuReviews[cols_to_concat] = df_RuReviews[cols_to_concat].fillna("").astype(str)
        df_RuReviews['text_result'] = df_RuReviews[cols_to_concat].agg(" ".join, axis=1)
        df_RuReviews.to_csv("df_RuReviews.csv", index=False)
        corpus = df_RuReviews['text_result'].tolist()[:50000]

        # нормализация, токенизация и лемматизация корпуса
        stop_words = self.stop_words
        normalized_corpus = [self.Normalization(text, stop_words) for text in tqdm(corpus)]
        with open("normalized_corpus_RuReviews.txt", "w", encoding="utf-8") as f:
            for line in normalized_corpus:
                f.write(line + "\n")

        lemmatized_corpus = [self.Tokenization_Lemmatization(text) for text in tqdm(normalized_corpus)]
        lemmatized_corpus = [" ".join(tokens) for tokens in lemmatized_corpus]

        with open("lemmatized_corpus_RuReviews.txt", "w", encoding="utf-8") as f:
            for line in lemmatized_corpus:
                f.write(line + "\n")

        return lemmatized_corpus