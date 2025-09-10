from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

'''
train_save_Tfidf
1) Запись в бинарный файл обученного на корпусе TfIdf
2) Запись в бинарный файл словаря с весами из TfIdf
'''
def train_save_Tfidf(full_corpus, vectorizer_path = 'vectorizer.pkl', tfidf_dict_path = 'tfidf_dict.pkl'):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(full_corpus)
    tfidf_dict = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    with open(tfidf_dict_path, 'wb') as f:
        pickle.dump(tfidf_dict, f)

    print(f"Сохранено: {vectorizer_path}, {tfidf_dict_path}")

'''
load_Tfidf
1) Загрузка обученного на корпусе TfIdf
2) Загрузка словаря с весами из TfIdf
'''
def load_Tfidf():

    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    with open('tfidf_dict.pkl', 'rb') as f:
        tfidf_dict = pickle.load(f)

    return vectorizer, tfidf_dict