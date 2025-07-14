import numpy as np
from datasets import load_dataset
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
from keras.callbacks import EarlyStopping
# Modelo : https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased
# Dataset: https://huggingface.co/datasets/Nicky0007/titulos_noticias_rcn_clasificadas

# 1. Cargar dataset
dataset = load_dataset("Nicky0007/titulos_noticias_rcn_clasificadas")

categorias = sorted(set(dataset["train"]["label"]))
categoria_to_index = {nombre: idx for idx, nombre in enumerate(categorias)}

def convertir_label(ejemplo):
    ejemplo["label"] = categoria_to_index[ejemplo["label"]]
    return ejemplo

dataset = dataset.map(convertir_label)

# 2. Tokenizar
modelo_id = "dccuchile/bert-base-spanish-wwm-uncased"
tokenizer = AutoTokenizer.from_pretrained(modelo_id)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

tokenized = dataset.map(tokenize)
tokenized.set_format("tensorflow", columns=["input_ids", "attention_mask", "label"])

# 3. Convertir a tensores de TF
def to_tf_dataset(ds):
    input_ids = np.array([x.numpy() if hasattr(x, 'numpy') else x for x in ds["input_ids"]])
    attention_mask = np.array([x.numpy() if hasattr(x, 'numpy') else x for x in ds["attention_mask"]])
    labels = np.array([x.numpy() if hasattr(x, 'numpy') else x for x in ds["label"]])

    return tf.data.Dataset.from_tensor_slices((
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        labels
    )).shuffle(500).batch(16)

train_ds = to_tf_dataset(tokenized["train"])
val_ds = to_tf_dataset(tokenized["test"])

# 4. Cargar modelo en TensorFlow
num_labels = len(set(dataset["train"]["label"]))
model = TFAutoModelForSequenceClassification.from_pretrained(modelo_id, num_labels=num_labels)

# 5. Compilar modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor='val_loss',     # métrica que se evalúa para detener
    patience=2,             # número de épocas sin mejora antes de parar
    restore_best_weights=True,  # restaura los pesos con mejor val_loss
    verbose=1
)

# 6. Entrenar modelo
model.fit(train_ds, validation_data=val_ds, epochs=5)

# 7. Guardar modelo
model.save_pretrained("beto_noticias_tf")
tokenizer.save_pretrained("beto_noticias_tf")