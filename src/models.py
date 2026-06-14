from __future__ import annotations

from typing import Optional

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Embedding, GlobalMaxPool1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class BaselineModel:
    """TF-IDF + logistic regression baseline for comparison."""

    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("lr", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        self.grid_search: Optional[GridSearchCV] = None

    def fit(self, X, y):
        y_series = pd.Series(y)
        class_counts = y_series.value_counts()
        valid_classes = class_counts[class_counts >= 3].index

        if len(valid_classes) == 0:
            X_filtered = X
            y_filtered = y
            cv_strategy = KFold(n_splits=3, shuffle=True, random_state=42)
        else:
            mask = y_series.isin(valid_classes)
            if hasattr(X, "iloc"):
                X_filtered = X[mask]
            else:
                X_filtered = [item for idx, item in enumerate(X) if mask.iloc[idx]]

            y_filtered = y_series[mask].to_numpy()
            cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        param_grid = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [3, 5],
            "tfidf__max_df": [0.7, 0.85],
            "lr__C": [0.1, 1.0, 10.0],
            "lr__penalty": ["l2"],
            "lr__class_weight": ["balanced", None],
        }
        # cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        self.grid_search = GridSearchCV(
            self.pipeline,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=0,
        )
        # self.grid_search.fit(X, y)
        self.grid_search.fit(X_filtered, y_filtered)
        self.pipeline = self.grid_search.best_estimator_
        return self.grid_search

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1_macro": f1_score(y, y_pred, average="macro"),
            "roc_auc": roc_auc_score(y, y_pred),
            "report": classification_report(y, y_pred),
        }

    def save(self, path: str):
        joblib.dump(self.pipeline, path)

    def load(self, path: str):
        self.pipeline = joblib.load(path)


class DeepLearningModel:
    """Compact CNN classifier then convert to TFLite for mobile deployment."""

    def __init__(
        self, max_vocab: int = 10000, max_len: int = 512, embedding_dim: int = 128
    ):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=self.max_vocab, oov_token="<OOV>")
        self.model = None

    def fit_tokenizer(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def texts_to_padded_sequences(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_len, padding="post")

    def build_model(self):
        model = Sequential(
            [
                tf.keras.Input(shape=(self.max_len,), dtype=tf.int32),
                Embedding(input_dim=self.max_vocab, output_dim=self.embedding_dim),
                Conv1D(128, 3, activation="relu", padding="same"),
                Conv1D(128, 5, activation="relu", padding="same"),
                Conv1D(64, 7, activation="relu", padding="same"),
                GlobalMaxPool1D(),
                Dense(32, activation="relu"),
                Dropout(0.3),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        self.model = model
        return model

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs: int = 20,
        batch_size: int = 64,
        class_weight=None,
    ):
        if self.model is None:
            self.build_model()
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=5,
            restore_best_weights=True
        )
        return self.model.fit(
            X_train,
            y_train,
            validation_data=(
                (X_val, y_val) if X_val is not None and y_val is not None else None
            ),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            verbose=0,
            callbacks=[early_stop]
        )

    def predict_proba(self, X):
        return self.model.predict(X, verbose=0).flatten()

    def save_tflite(self, tflite_path: str):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as file_handle:
            file_handle.write(tflite_model)

    def evaluate_tflite(self, X, y, model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        tflite_probs = []
        for i in range(len(X)):
            input_data = np.expand_dims(X[i], axis=0).astype(np.int32)
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]["index"])
            tflite_probs.append(output_data[0][0])

        tflite_probs = np.array(tflite_probs)

        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_f1 = 0

        for t in thresholds:
            preds = (tflite_probs > t).astype(int)
            score = f1_score(y, preds)

            if score > best_f1:
                best_f1 = score
                best_threshold = t

        tflite_preds = (tflite_probs >= best_threshold).astype(int)
        tflite_size = (os.path.getsize(model_path) / (1024 * 1024))

        return {
            "size": tflite_size,
            "accuracy": accuracy_score(y, tflite_preds),
            "f1_macro": f1_score(y, tflite_preds, average="macro"),
            "roc_auc": roc_auc_score(y, tflite_probs),
            "report": classification_report(y, tflite_preds),
        }