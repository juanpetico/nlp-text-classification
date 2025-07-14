import tkinter as tk
from tkinter import messagebox
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import numpy as np

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

# 2. Función para clasificar el texto ingresado
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

# 3. Interfaz con Tkinter 
window = tk.Tk()
window.title("Clasificador de Noticias")

tk.Label(window, text="Ingresa una noticia:").pack(pady=5)
entry = tk.Entry(window, width=60)
entry.pack(pady=5)

tk.Button(window, text="Predecir categoría", command=predic_entry).pack(pady=5)

result = tk.StringVar()
tk.Label(window, textvariable=result, font=("Helvetica", 12), fg="blue").pack(pady=10)

window.mainloop()
