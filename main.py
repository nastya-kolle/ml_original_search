from Text_Preprocessing import Text_Preprocessing
from Dataset_Preprocessing import Dataset_Preprocessing
import os

def main():

    text_processor = Text_Preprocessing()
    # input string
    string = "  Это   примерный ТЕКСТ, содержащий  цифры 12.3 100% и знаки препинания, т.е. также адресс и т.д. adidas!!!  "

    #нормализация
    normalized_string = text_processor.Normalization(string, text_processor.stop_words)
    print("normalized_string")
    print(normalized_string)

    #токенизация и лемматизация
    lemmatized_string = text_processor.Tokenization_Lemmatization(normalized_string)
    print("lemmatized_string")
    print(lemmatized_string)

    #NER
    string_1 = "Президент России Владимир Путин встретился с представителями ООН в Москве."
    NER = text_processor.Named_Entity_Recognition(string_1)
    print("NER")
    print(NER)

    #Эмбеддинг с помощью среднего
    embedding_mean = text_processor.Embedding_Mean(lemmatized_string)
    print("embedding_mean")
    print(embedding_mean)

    # Эмбеддинг с помощью взвешенного среднего
    embedding_tfidf = text_processor.Embedding_Tfidf(lemmatized_string)
    print("embedding_tfidf")
    print(embedding_tfidf)

    dataset_processor = Dataset_Preprocessing()

    #Загрузка датасета
    df_ozon_train, df_ozon_test = dataset_processor.load_dataset()

    #Предобработка обучающей и тестовой выборки
    df_ozon_train = dataset_processor.preprocess_dataset(df_ozon_train)
    df_ozon_test = dataset_processor.preprocess_dataset(df_ozon_test)

    # Анализ данных
    """
    dataset_processor.rating_1_count_analitics(df_ozon_train)
    dataset_processor.rating_2_count_analitics(df_ozon_train)
    dataset_processor.rating_3_count_analitics(df_ozon_train)
    dataset_processor.rating_4_count_analitics(df_ozon_train)
    dataset_processor.rating_5_count_analitics(df_ozon_train)
    dataset_processor.comments_published_count_analitics(df_ozon_train)
    dataset_processor.photos_published_count_analitics(df_ozon_train)
    dataset_processor.videos_published_count_analitics(df_ozon_train)
    dataset_processor.PriceDiscounted_analitics(df_ozon_train)
    dataset_processor.item_count_fake_returns7_analitics(df_ozon_train)
    dataset_processor.item_count_fake_returns30_analitics(df_ozon_train)
    dataset_processor.item_count_fake_returns90_analitics(df_ozon_train)
    dataset_processor.item_count_sales7_analitics(df_ozon_train)
    dataset_processor.item_count_sales30_analitics(df_ozon_train)
    dataset_processor.item_count_sales90_analitics(df_ozon_train)
    dataset_processor.item_count_returns7_analitics(df_ozon_train)
    dataset_processor.item_count_returns30_analitics(df_ozon_train)
    dataset_processor.item_count_returns90_analitics(df_ozon_train)
    dataset_processor.GmvTotal7_analitics(df_ozon_train)
    dataset_processor.GmvTotal30_analitics(df_ozon_train)
    dataset_processor.GmvTotal90_analitics(df_ozon_train)
    dataset_processor.ExemplarAcceptedCountTotal7_analitics(df_ozon_train)
    dataset_processor.ExemplarAcceptedCountTotal30_analitics(df_ozon_train)
    dataset_processor.ExemplarAcceptedCountTotal90_analitics(df_ozon_train)
    dataset_processor.OrderAcceptedCountTotal7_analitics(df_ozon_train)
    dataset_processor.OrderAcceptedCountTotal30_analitics(df_ozon_train)
    dataset_processor.OrderAcceptedCountTotal90_analitics(df_ozon_train)
    dataset_processor.ExemplarReturnedCountTotal7_analitics(df_ozon_train)
    dataset_processor.ExemplarReturnedCountTotal30_analitics(df_ozon_train)
    dataset_processor.ExemplarReturnedCountTotal90_analitics(df_ozon_train)
    dataset_processor.ExemplarReturnedValueTotal7_analitics(df_ozon_train)
    dataset_processor.ExemplarReturnedValueTotal30_analitics(df_ozon_train)
    dataset_processor.ExemplarReturnedValueTotal90_analitics(df_ozon_train)
    dataset_processor.ItemVarietyCount_analitics(df_ozon_train)
    dataset_processor.ItemAvailableCount_analitics(df_ozon_train)
    dataset_processor.seller_time_alive_analitics(df_ozon_train)
    dataset_processor.item_time_alive_analitics(df_ozon_train)

    dataset_processor.resolution_analitics(df_ozon_train)
    dataset_processor.brand_name_analitics(df_ozon_train)
    dataset_processor.SellerID_analitics(df_ozon_train)
    dataset_processor.CommercialCategory_analitics(df_ozon_train)
    """

    #Создания фич для дальнейшей загрузки в модель для обучающей и тестовой выборок
    df_ozon_train = dataset_processor.feature_engineering_returns(df_ozon_train)
    df_ozon_train = dataset_processor.feature_engineering_ratings(df_ozon_train)
    df_ozon_train = dataset_processor.feature_engineering_sales(df_ozon_train)
    df_ozon_train = dataset_processor.feature_engineering_seller(df_ozon_train)
    df_ozon_train = dataset_processor.feature_engineering_availability_variety(df_ozon_train)
    df_ozon_train = dataset_processor.feature_engineering_price(df_ozon_train)
    df_ozon_train = dataset_processor.feature_engineering_brand_seller(df_ozon_train)

    df_ozon_test = dataset_processor.feature_engineering_returns(df_ozon_test)
    df_ozon_test = dataset_processor.feature_engineering_ratings(df_ozon_test)
    df_ozon_test = dataset_processor.feature_engineering_sales(df_ozon_test)
    df_ozon_test = dataset_processor.feature_engineering_seller(df_ozon_test)
    df_ozon_test = dataset_processor.feature_engineering_availability_variety(df_ozon_test)
    df_ozon_test = dataset_processor.feature_engineering_price(df_ozon_test)
    df_ozon_test = dataset_processor.feature_engineering_brand_seller(df_ozon_test)


    #Превращение всех текстовых переменных в эмбеддинги и создание необходимых фич для обучаюшей и тестовой выборок
    df_ozon_train = dataset_processor.name_fraud_detection(df_ozon_train)
    df_ozon_train = dataset_processor.brand_fraud_detection(df_ozon_train)
    df_ozon_train = dataset_processor.description_fraud_detection(df_ozon_train)

    df_ozon_test = dataset_processor.name_fraud_detection(df_ozon_test)
    df_ozon_test = dataset_processor.brand_fraud_detection(df_ozon_test)
    df_ozon_test = dataset_processor.description_fraud_detection(df_ozon_test)

    #Добавление NER фич
    df_ozon_train = dataset_processor.NER(df_ozon_train)
    df_ozon_test = dataset_processor.NER(df_ozon_test)

    #Обучение модели в первый раз
    df_ozon_train = dataset_processor.train_model(df_ozon_train)

    #Загрузка уже обученной модели и pca
    if not os.path.exists('pca_name.pkl'):
        raise FileNotFoundError('File pca_name.pkl is not found')
    if not os.path.exists('pca_desc.pkl'):
        raise FileNotFoundError('File pca_desc.pkl is not found')
    if not os.path.exists('pca_brand.pkl'):
        raise FileNotFoundError('File pca_brand.pkl is not found')
    else:
        df_ozon_test = dataset_processor.prepare_features(df_ozon_test)

    #Предсказание
    print(df_ozon_test.columns)
    Y_pred, Y_proba = dataset_processor.predict(df_ozon_test)

    print("Y_pred ", Y_pred)
    print("Y_proba ", Y_proba)

if __name__ == '__main__':
    main()
