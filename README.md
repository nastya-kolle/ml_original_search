# ML Original Search — Counterfeit Product Detection

A machine learning pipeline built for Ozon that predicts whether a marketplace listing is **counterfeit ("контрафакт")** or an original product, using a mix of text, seller, and transaction-behavior features.

## What the Model Predicts

Given a product listing (name, brand, description, category) together with seller/marketplace behavioral signals (sales, returns, ratings, GMV, price, catalog variety, etc.), the pipeline predicts `resolution` — a binary label indicating whether an operator/moderator would flag the item as counterfeit.

## Pipeline Overview

1. **Text preprocessing** (`Text_Preprocessing.py`) — normalization, tokenization/lemmatization (pymorphy2, razdel), spell-checking, stop-word removal, and Named Entity handling for Russian product text. Produces embeddings via FastText (mean pooling) and TF-IDF weighting.
2. **Dataset preprocessing & feature engineering** (`Dataset_Preprocessing.py`) — the core module:
   - Loads and cleans the Ozon counterfeit train/test CSVs
   - Exploratory analytics per feature (ratings, returns, sales, GMV, seller/item lifetime, price, category, brand, etc.)
   - Feature engineering: return rates, fake-return ratios, GMV/order ratios, price vs. category/brand medians, seller-level aggregates, catalog variety ratios, and more
   - **Name/brand/description fraud signals**: text embeddings, clustering (HDBSCAN/DBSCAN + UMAP) to flag outlier names/descriptions, and cosine similarity between product name and known brand vectors
   - **NER-based features** (`NER.py`): extracts entities (name, price, brand, category, country) from titles/descriptions with a fine-tuned Flair NER model, then derives consistency features (e.g. does the brand mentioned in the description match the declared brand?)
3. **Dimensionality reduction** — PCA (30 components) applied to the name/description/brand embeddings before modeling.
4. **Modeling**:
   - **CatBoostClassifier** — the main counterfeit classifier, trained on the full engineered feature set (numeric + categorical), saved as `CatBoost_model.cbm`
   - **XGBoost** (`xgboost_utils.py`, `xgboost_preparation.py`) — a secondary multi-class model that predicts product category from name embeddings
   - **TF-IDF** (`tfidf_utils.py`, `tfidf_preparation.py`) — a vectorizer trained on a Wildberries/RuReviews text corpus, used for embedding lemmatized text
5. **Inference** (`main.py`) — loads the trained PCA transforms and model, prepares features for new data, and outputs predicted labels and probabilities.

## Repository Structure

```
ml_original_search/
├── main.py                     # End-to-end pipeline entry point (train + predict)
├── Text_Preprocessing.py       # Text normalization, lemmatization, NER, embeddings
├── Dataset_Preprocessing.py    # Feature engineering, fraud detection, model training/inference
├── NER.py                      # Flair-based NER model wrapper (train/load/extract entities)
├── model_loader.py             # Loads the FastText model (cc.ru.300.bin) once, shared across modules
├── tfidf_utils.py               # Train/save/load a TF-IDF vectorizer
├── tfidf_preparation.py        # One-off script: builds the text corpus and trains TF-IDF
├── xgboost_utils.py            # Train/save/load the XGBoost category classifier
├── xgboost_preparation.py      # One-off script: trains and saves the XGBoost model
├── decompress.py               # Decompresses Wildberries products/feedbacks `.zst` archives
└── dataset_process_copy.py, draft.py  # Working/exploratory copies of the preprocessing logic
```

## Data

The pipeline expects (not included in the repo):
- `ml_ozon_сounterfeit_train.csv` / `ml_ozon_сounterfeit_test.csv` — the labeled Ozon counterfeit dataset (product metadata, ratings, returns, sales, GMV, seller info)
- A Wildberries products/feedbacks corpus (`nyuuzyou/wb-products`, `nyuuzyou/wb-feedbacks`) for training TF-IDF and NER, decompressed via `decompress.py`
- `cc.ru.300.bin` — pretrained Russian FastText embeddings

## Requirements

Key Python dependencies (see imports for the full list):
`pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `catboost`, `hdbscan`, `umap-learn`, `fasttext`, `flair`, `natasha`, `pymorphy2`, `razdel`, `pyspellchecker`, `rapidfuzz`, `transliterate`, `nltk`, `beautifulsoup4`, `num2words`, `ijson`, `datasets`, `zstandard`, `tqdm`, `joblib`

## Usage

```bash
# 1. Decompress the raw Wildberries corpus (one-off)
python decompress.py

# 2. Build the text corpus and train TF-IDF (one-off)
python tfidf_preparation.py

# 3. Train the XGBoost category classifier (one-off)
python xgboost_preparation.py

# 4. Run the full pipeline: preprocessing, feature engineering, CatBoost training, and prediction
python main.py
```

## Notes

- `NER.py` currently loads its trained model from a hard-coded local path (`C:/Nastya/ozon_py/my_project/models/my_ner_model/final-model.pt`). Update this path before running on another machine.
- `draft.py` and `dataset_process_copy.py` appear to be exploratory/working copies of the main preprocessing logic and are not part of the primary pipeline.
