from Text_Preprocessing import Text_Preprocessing
from tfidf_utils import train_save_Tfidf
'''
Единоразовый запуск загрузки корпуса и его подача в TfIdf
'''
def main():
    processor = Text_Preprocessing()

    #загрузка корпуса
    corpus_Wildberries = processor.Corpus_Wildberries()
    corpus_RuReviews = processor.Corpus_RuReviews()
    full_corpus = corpus_Wildberries + corpus_RuReviews

    if not full_corpus:
        raise ValueError("Corpus is empty! Проверьте источники данных.")

    # сохранение корпуса в текстовый файл
    with open("full_corpus.txt", "w", encoding="utf-8") as f:
        for line in full_corpus:
            f.write(line+'\n')

    #загрузка и дообучение TfIdf
    train_save_Tfidf(full_corpus)

if __name__ == '__main__':
   main()