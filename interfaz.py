import tkinter as tk
from tkinter import messagebox
from datasets import load_dataset
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Cargar modelo y tokenizer guardados
model = TFAutoModelForSequenceClassification.from_pretrained("./beto_noticias_tf")
tokenizer = AutoTokenizer.from_pretrained("./beto_noticias_tf")

index_to_categoria = {
    0: "economia",
    1: "deportes",
    2: "colombia",
    3: "salud",
    4: "tecnologia"
}

# Cargar el dataset para la matriz confusión
dataset = load_dataset("Nicky0007/titulos_noticias_rcn_clasificadas")
categories = sorted(set(dataset["train"]["label"]))
categoriesIdx = {name: idx for idx, name in enumerate(categories)}

def convertLabel(example):
    example["label"] = categoriesIdx[example["label"]]
    return example

dataset = dataset.map(convertLabel)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

tokenized = dataset["test"].map(tokenize)
tokenized.set_format("tensorflow", columns=["input_ids", "attention_mask", "label"])

def to_tf_dataset(ds):
    input_ids = np.array([x.numpy() if hasattr(x, 'numpy') else x for x in ds["input_ids"]])
    attention_mask = np.array([x.numpy() if hasattr(x, 'numpy') else x for x in ds["attention_mask"]])
    labels = np.array([x.numpy() if hasattr(x, 'numpy') else x for x in ds["label"]])

    return tf.data.Dataset.from_tensor_slices(({
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }, labels)).batch(16)

test_ds = to_tf_dataset(tokenized)

# 2. Función para mostrar la matriz de confusión
def mostrar_matriz_confusion():
    y_true = []
    y_pred = []

    for batch in test_ds:
        inputs, labels = batch
        logits = model.predict(inputs).logits
        preds = tf.argmax(logits, axis=1).numpy()

        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(index_to_categoria.values())
    )
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Matriz de Confusión - Clasificador BETO Noticias")
    plt.tight_layout()
    plt.show()

# 3. Función para clasificar el texto ingresado
def predic_entry():
    message = entry.get()
    if not message.strip():
        messagebox.showwarning("Error", "Ingresa un mensaje primero.")
        return

    # Tokenizar el mensaje
    message_tokenized = tokenizer(
        message,
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=64
    )

    # Obtener logits y 
    logits = model(message_tokenized)[0]
    pred = tf.argmax(logits, axis=1).numpy()[0]
    conf = tf.nn.softmax(logits, axis=1).numpy()[0][pred]

    result.set(f"Categoría: {index_to_categoria[pred]} ({conf*100:.2f}%)")

# 4. Interfaz con Tkinter 
window = tk.Tk()
window.title("Clasificador de Noticias")

tk.Label(window, text="Ingresa una noticia:").pack(pady=5)
entry = tk.Entry(window, width=60)
entry.pack(pady=5)

tk.Button(window, text="Predecir categoría", command=predic_entry).pack(pady=5)

result = tk.StringVar()
tk.Label(window, textvariable=result, font=("Helvetica", 12), fg="blue").pack(pady=10)

tk.Button(window, text="Ver matriz de confusión", command=mostrar_matriz_confusion).pack(pady=5)

window.mainloop()
