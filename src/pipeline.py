from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from .data_cleaner import TextCleaner


@dataclass
class DatasetBuilder:
    """Load source files and prepare a unified dataset for training."""

    data_dir: str = "dataset"

    def load_excel(self, filename: str) -> pd.DataFrame:
        return pd.read_excel(os.path.join(self.data_dir, filename))

    def load_sample_dataset(
        self, filename: str = "sample_dataset.xlsx"
    ) -> pd.DataFrame:
        """Load the small sample dataset used for demos."""
        return self.load_excel(filename)

    def build_from_sources(
        self, sources: Dict[str, str], label_column: str = "hoax"
    ) -> pd.DataFrame:
        frames = []
        for filename, text_column in sources.items():
            frame = self.load_excel(filename).copy()
            if "FullText" in text_column:
                frame["FullText"] = frame[text_column].apply(
                    TextCleaner.extract_narasi_only
                )
            if "clean_text" not in frame.columns:
                frame["clean_text"] = frame[text_column].apply(
                    TextCleaner.clean_text_basic
                )
            if "cnn" in filename.lower():
                frame["clean_text"] = frame["clean_text"].apply(
                    TextCleaner.clean_cnn
                )
            elif "kompas" in filename.lower():
                frame["clean_text"] = frame["clean_text"].apply(
                    TextCleaner.clean_kompas
                )
            elif "tempo" in filename.lower():
                frame["clean_text"] = frame["clean_text"].apply(
                    TextCleaner.clean_tempo
                )
            elif "hoax" in filename.lower():
                frame["clean_text"] = frame["clean_text"].apply(
                    TextCleaner.clean_hoax
                )
            frames.append(frame)

        dataset = pd.concat(frames, ignore_index=True)
        dataset = dataset.dropna(subset=["clean_text"])
        dataset = dataset.drop_duplicates(subset=["clean_text"])
        if label_column not in dataset.columns:
            raise ValueError(
                f"Label column '{label_column}' was not found in the combined dataset."
            )
        return dataset.sample(frac=1, random_state=42).reset_index(drop=True)


@dataclass
class FeatureEngineer:
    """Simple wrapper for splitting text data into train and validation sets."""

    test_size: float = 0.2
    random_state: int = 42

    def split(self, texts: Sequence[str], labels: Sequence[int]):
        return train_test_split(
            texts,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels,
        )


def train_test_split_stratified(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
