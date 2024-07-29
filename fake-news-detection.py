import pandas as pd
import numpy as np
import re
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


real1 = pd.read_excel("dataset/dataset_cnn_10k_cleaned.xlsx")
real2 = pd.read_excel("dataset/dataset_kompas_4k_cleaned.xlsx")
real3 = pd.read_excel("dataset/dataset_tempo_6k_cleaned.xlsx")
fake = pd.read_excel("dataset/dataset_turnbackhoax_10_cleaned.xlsx")


real1.drop(
    ["Title", "Timestamp", "FullText", "Tags", "Author", "Url"], axis=1, inplace=True
)
real2.drop(
    ["Title", "Timestamp", "FullText", "Tags", "Author", "Url"], axis=1, inplace=True
)
real3.drop(
    ["Title", "Timestamp", "FullText", "Tags", "Author", "Url"], axis=1, inplace=True
)
fake.drop(
    ["Title", "Timestamp", "FullText", "Tags", "Author", "Url", "politik", "Narasi"],
    axis=1,
    inplace=True,
)

dataset = pd.concat([real1, real2, real3, fake]).sample(frac=1)

dataset.dropna(subset=["text_new"], inplace=True)

dataset["text_new"] = dataset["text_new"].apply(lambda x: re.sub("\W+", " ", x))

sentences = dataset["text_new"]
labels = dataset["hoax"]

trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"

training_size = 19000
vocab_size = 5000
max_length = 300
embedding_dim = 100

training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]

training_labels = labels[:training_size]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(
    num_words=vocab_size,
    oov_token=oov_tok,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{"}~\t\n',
)
tokenizer.fit_on_texts(training_sentences)

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(
    training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(
    testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
)

training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
num_epochs = 10

history = model.fit(
    training_padded,
    training_labels,
    epochs=num_epochs,
    validation_data=(testing_padded, testing_labels),
    verbose=2,
    callbacks=[early_stopping],
)

# model.summary()


def test_model(text):
    testing_text = [text]
    testing_text = tokenizer.texts_to_sequences(testing_text)
    testing_text = pad_sequences(testing_text, maxlen=max_length)
    prediction = model.predict(testing_text)
    news = ""
    if prediction >= 0.5:
        news = "Fake News"
    else:
        news = "Real News"
    return news


def plot_loss_acc(history):
    """Plots the training and validation loss and accuracy from a history object"""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(acc))

    plt.plot(epochs, acc, "g", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, "r", label="Training Loss")
    plt.plot(epochs, val_loss, "b", label="Validation Loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.show()


# plot_loss_acc(history)

# test_model("Rumah Meledak Usai Seorang Pria Semprotkan Insektisida untuk Usir Kecoak")
# testing_text = [
#     "Rumah Meledak Usai Seorang Pria Semprotkan Insektisida untuk Usir Kecoak"
# ]
# testing_text = tokenizer.texts_to_sequences(testing_text)
# testing_text = pad_sequences(testing_text, maxlen=max_length)
# model.predict(testing_text)
