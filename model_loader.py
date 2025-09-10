import fasttext
print("Загружается FastText модель (может занять время)...")
model = fasttext.load_model("cc.ru.300.bin")
print("FastText модель загружена.")
