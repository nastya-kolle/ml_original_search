from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Sentence
import os
"""
Активация виртуальной среды
.\env_flair\scripts\activate

Выберите Flair если:
Нужна легкая кастомизация тегов
Хотите экспериментировать с архитектурой
Нужен transfer learning
Работаете с разными типами эмбеддингов
Цените простоту дообучения


крч, в последний раз напомню:
NAME ~ 200
PRICE ~ 200
CATEGORY ~ 200
BRAND ~ 200
COUNTRY ~ 100
"""
class NER_Preprocessing():

    def __init__(self, load_model=True):
        self.model_path = 'C:/Nastya/ozon_py/my_project/models/my_ner_model/final-model.pt'
        self.trained_model = self.load_NER_model() if load_model else None

    def train_model(self):
        # 1. Укажите путь к вашим данным
        data_folder = 'C:/Nastya/ozon_py/my_project/data' # Папка где лежат train.txt, test.txt, valid.txt

        #Проверка существования файлов
        required_files = ['train.txt', 'valid.txt', 'test.txt']
        for file in required_files:
            file_path = os.path.join(data_folder, file)
            if not os.path.exists(file_path):
                print(f"❌ Файл {file_path} не найден!")
                return
        print("✅ Все файлы данных найдены")

        # 2. Загрузка корпуса данных
        corpus = ColumnCorpus(
            data_folder,
            column_format={0: 'text', 1: 'ner'}, # Ваш формат: токен + тег
            train_file='train.txt',
            test_file='test.txt',
            dev_file='valid.txt'
        )

        # 3. Посмотрим статистику данных
        print("Размер корпуса: ")
        print(f"Обучающая выборка: {len(corpus.train)} примеров")
        print(f"Валидационная выборка: {len(corpus.dev)} примеров")
        print(f"Тестовая выборка: {len(corpus.test)} примеров")

        # 4. Создаем словарь тегов
        label_dict = corpus.make_label_dictionary(label_type='ner')
        print("\nОбнаруженные теги: ")
        for label in label_dict.get_items():
            print(f" {label}")

        # 5. Создаем эмбеддинги (лучшие для русского)
        ru_bert = TransformerWordEmbeddings(
            model='DeepPavlov/rubert-base-cased',
            layers='-1',
            subtoken_pooling='first',
            fine_tune=True, # Разрешаем дообучение эмбеддингов
            use_context=True,
        )

        # 6. Создаем модель NER
        tagger = SequenceTagger(
            hidden_size=256,
            embeddings=ru_bert,
            tag_dictionary=label_dict,
            tag_type='ner',
            use_crf=True, # CRF улучшает качество NER
            use_rnn=True,
            rnn_layers=1,
            reproject_embeddings=False,
        )

        # 7. Инициализируем тренер
        trainer = ModelTrainer(tagger, corpus)

        # 8. Создаем папку для модели
        model_path = 'C:/Nastya/ozon_py/my_project/models/my_ner_model'
        os.makedirs(model_path, exist_ok=True)

        # 8. Запускаем обучение
        print("\n🚀 Начинаем обучение...")
        trainer.train(
            model_path, # Папка для сохранения модели
            learning_rate=0.1,
            mini_batch_size=16,
            max_epochs=5,
            train_with_dev=True, # Используем dev set для валидации
        )

        print(f"✅ Обучение завершено! Модель сохранена в: {model_path}")

    def load_NER_model(self):

        if not os.path.exists(self.model_path):
            raise FileNotFoundError("❌ Модель не найдена! Сначала запустите обучение.")

        try:
            # Загрузка обученной модели
            trained_model = SequenceTagger.load(self.model_path)
            print("✅ Модель успешно загружена")
            return trained_model
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return None

    def extract_NER_entities(self, text, verbose=False):
        try:
            if not isinstance(text, str) or text.strip() == "":
                return []

            sentence = Sentence(text)
            self.trained_model.predict(sentence)

            entities = []
            for entity in sentence.get_spans('ner'):
                if verbose:
                    print(f"   '{entity.text}' -> {entity.tag} (уверенность: {entity.score:.2f})")
                entities.append({
                    "text": entity.text,
                    "tag": entity.tag,
                    "score": round(entity.score, 3)})
            return entities
        except Exception as e:
            print(f"❌ Ошибка при извлечении сущностей: {e}")
            import traceback
            traceback.print_exc()
            return []
