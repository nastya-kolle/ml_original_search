import os
from Dataset_Preprocessing import prepare_sample_df
from xgboost_utils import train_XGBoost, save_XGBoost, load_XGBoost
model_path = 'xgb_classifier.pkl'
label_encoder_path = 'label_encoder.pkl'
'''
Единоразовый запуск загрузки и обучения XGBoost
'''
def main():
    force_train = True
    if os.path.exists(model_path) and not force_train:
        print(f"[INFO] Модель уже существует: {model_path}")
        #model, label_encoder = load_XGBoost()
    else:
        print("[INFO] Обучаем новую модель...")
        sample_df = prepare_sample_df()
        model, label_encoder = train_XGBoost(sample_df)

        # загрузка и дообучение TfIdf
        save_XGBoost(model, label_encoder)
        print(f"[INFO] Модель сохранена в: {model_path}")
        print(f"[INFO] LabelEncoder сохранен в: {label_encoder_path}")

if __name__ == '__main__':
   main()
