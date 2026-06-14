from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from src import BaselineModel, DeepLearningModel, DatasetBuilder, FeatureEngineer

DATA_DIR = Path("dataset")
MODELS_DIR = Path("models")


def prepare_sample_fallback() -> pd.DataFrame:
    sample_path = DATA_DIR / "sample_dataset.xlsx"
    if not sample_path.exists():
        raise FileNotFoundError("sample_dataset.xlsx was not found in dataset/")
    return pd.read_excel(sample_path)


def train_and_export():
    builder = DatasetBuilder(data_dir=str(DATA_DIR))
    dataset = builder.load_sample_dataset("sample_dataset.xlsx")
    # dataset = builder.build_from_sources(
    #     {
    #         "cnn_cleaned.xlsx": "text_new",
    #         "kompas_cleaned.xlsx": "text_new",
    #         "tempo_cleaned.xlsx": "text_new",
    #         "turnbackhoax_cleaned.xlsx": "FullText",
    #     },
    #     label_column="hoax",
    # )

    if "clean_text" not in dataset.columns:
        dataset["clean_text"] = dataset["text_new"].fillna("").astype(str)

    if "hoax" in dataset.columns:
        y = dataset["hoax"]
    elif "labels" in dataset.columns:
        y = dataset["labels"]
    else:
        raise ValueError(
            "The sample dataset must include a label column named 'hoax' or 'labels'."
        )

    X = dataset["clean_text"]

    feature_engineer = FeatureEngineer(test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = feature_engineer.split(X, y)

    baseline = BaselineModel()
    baseline.fit(X_train, y_train)
    baseline_metrics = baseline.evaluate(X_val, y_val)

    MODELS_DIR.mkdir(exist_ok=True)
    baseline.save(MODELS_DIR / "baseline.pkl")

    deep_model = DeepLearningModel()
    deep_model.fit_tokenizer(X_train.tolist())
    X_train_pad = deep_model.texts_to_padded_sequences(X_train.tolist())
    X_val_pad = deep_model.texts_to_padded_sequences(X_val.tolist())

    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    class_weight_map = dict(enumerate(class_weights))

    deep_model.build_model()
    deep_model.fit(
        X_train_pad,
        y_train,
        X_val=X_val_pad,
        y_val=y_val,
        epochs=20,
        batch_size=64,
        class_weight=class_weight_map,
    )
    deep_model.model.save(MODELS_DIR / "HoaXGY_model.keras")
    deep_model.save_tflite(MODELS_DIR / "HoaXGY_model.tflite")

    print("TFLite metrics:")
    tflite_metrics = deep_model.evaluate_tflite(X_val_pad, y_val, MODELS_DIR / "HoaXGY_model.tflite")
    size = tflite_metrics["size"]
    accuracy = tflite_metrics["accuracy"]
    roc_auc = tflite_metrics["roc_auc"]
    print(f"Size: {size:.2f} MB")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}\n")
    print(tflite_metrics["report"])
    print(f"Keras model exported to {MODELS_DIR / 'HoaXGY_model.keras'}")
    print(f"TFLite model exported to {MODELS_DIR / 'HoaXGY_model.tflite'}")


if __name__ == "__main__":
    train_and_export()
