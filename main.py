import numpy as np
from datasets import load_dataset
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
from keras.callbacks import EarlyStopping
# Modelo : https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased
# Dataset: https://huggingface.co/datasets/Nicky0007/titulos_noticias_rcn_clasificadas

# 1. Cargar dataset
dataset = load_dataset("Nicky0007/titulos_noticias_rcn_clasificadas")

# 2. Mapear categorías a valores numéricos
categories = sorted(set(dataset["train"]["label"]))
categoriesIdx = {nombre: idx for idx, nombre in enumerate(categories)}

def convertLabel(example):
    example["label"] = categoriesIdx[example["label"]]
    return example

dataset = dataset.map(convertLabel)

# 3. Tokenizar
model_id = "dccuchile/bert-base-spanish-wwm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

tokenized = dataset.map(tokenize)
tokenized.set_format("tensorflow", columns=["input_ids", "attention_mask", "label"])

# 4. Convertir a tensores de TF
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
test_ds = to_tf_dataset(tokenized["test"])

# 5. Cargar modelo en TensorFlow
num_labels = len(set(dataset["train"]["label"]))
model = TFAutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)

# 6. Compilar modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor='val_loss',     
    patience=2,             
    restore_best_weights=True,  
    verbose=1
)

# 7. Entrenar modelo
model.fit(train_ds, validation_data=test_ds, epochs=5)

# 8. Guardar modelo
model.save_pretrained("beto_noticias_tf")
tokenizer.save_pretrained("beto_noticias_tf")