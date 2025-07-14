import tkinter as tk
from tkinter import messagebox
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import numpy as np

# === 1. Cargar modelo y tokenizer guardados ===
model = TFAutoModelForSequenceClassification.from_pretrained("./beto_noticias_tf")
tokenizer = AutoTokenizer.from_pretrained("./beto_noticias_tf")

# Mapear IDs a etiquetas reales (ajústalo con las tuyas)
index_to_categoria = {
    0: "economia",
    1: "deportes",
    2: "colombia",
    3: "salud",
    4: "tecnologia"
}

# === 2. Función para clasificar el texto ingresado ===
def predecir():
    mensaje = entry.get()
    if not mensaje.strip():
        messagebox.showwarning("Error", "Ingresa un mensaje primero.")
        return

    # Tokenizar el mensaje
    tokens = tokenizer(
        mensaje,
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=64
    )

    # Obtener logits y predicción
    logits = model(tokens)[0]
    pred = tf.argmax(logits, axis=1).numpy()[0]
    conf = tf.nn.softmax(logits, axis=1).numpy()[0][pred]

    resultado.set(f"Categoría: {index_to_categoria[pred]} ({conf*100:.2f}%)")

# === 3. Interfaz con Tkinter ===
ventana = tk.Tk()
ventana.title("Clasificador de Noticias")

tk.Label(ventana, text="Ingresa una noticia:").pack(pady=5)
entry = tk.Entry(ventana, width=60)
entry.pack(pady=5)

tk.Button(ventana, text="Predecir categoría", command=predecir).pack(pady=5)

resultado = tk.StringVar()
tk.Label(ventana, textvariable=resultado, font=("Helvetica", 12), fg="blue").pack(pady=10)

ventana.mainloop()
