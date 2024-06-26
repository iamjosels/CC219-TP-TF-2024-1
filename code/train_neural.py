import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pandas as pd
import pickle

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv('customer_support_tickets_clean.csv')

# Preprocesar los datos
desc = df['Processed_Description'][:1000].tolist()
labels = df['Ticket Type'][:1000].tolist()

tests = df['Processed_Description'][1000:1110].tolist()
test_labels = df['Ticket Type'][1000:1110].tolist()

# Paso 1: Tokenizar las oraciones
tokenizer = Tokenizer()
tokenizer.fit_on_texts(desc)
secuencias = tokenizer.texts_to_sequences(desc)
word_index = tokenizer.word_index

# Paso 2: Pad the sequences
max_length = max(len(seq) for seq in secuencias)
X = pad_sequences(secuencias, maxlen=max_length, padding='post')

# Paso 3: Convertir etiquetas a one-hot
label_encoder = LabelEncoder()
int_encoded = label_encoder.fit_transform(labels)
y = to_categorical(int_encoded)

# Crear el modelo
modelo = tf.keras.Sequential()
modelo.add(tf.keras.layers.Embedding(input_dim=len(word_index) + 1, output_dim=50, input_length=max_length))
modelo.add(tf.keras.layers.Flatten())
modelo.add(tf.keras.layers.Dense(units=128, activation='relu'))
modelo.add(tf.keras.layers.Dense(units=128, activation='relu'))
modelo.add(tf.keras.layers.Dense(units=len(label_encoder.classes_), activation='softmax'))

# Compilar el modelo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Resumen del modelo
modelo.summary()

# Entrenar el modelo
modelo.fit(X, y, epochs=10, batch_size=1)

# Guardar el modelo
modelo.save('nn_model.h5')

# Guardar el tokenizer y el label_encoder
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Guardar la longitud máxima
with open('max_length.pkl', 'wb') as f:
    pickle.dump(max_length, f)

# Función para predecir la categoría
def predecir_categoria(oracion):
    secuencia = tokenizer.texts_to_sequences([oracion])
    secuencia_pad = pad_sequences(secuencia, maxlen=max_length, padding='post')
    prediccion = modelo.predict(secuencia_pad)
    categoria_codificada = np.argmax(prediccion)
    categoria = label_encoder.inverse_transform([categoria_codificada])
    return categoria[0]

# Evaluar el modelo
mal = 0
bien = 0

for i in range(110):
    if predecir_categoria(tests[i]) != test_labels[i]:
        mal += 1
    else:
        bien += 1

print("MAL: ", mal)
print("BIEN: ", bien)
