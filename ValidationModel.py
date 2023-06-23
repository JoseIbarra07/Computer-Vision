import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Carga y preprocesamiento de datos
data_dir = "C:/Users/USER/Documents/Maestria/Primero/Visión Computacional/Proyecto/Gender Recognition/Validation"
train_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(data_dir, target_size=(100, 100), class_mode='categorical')

# Creación del modelo
model = tf.keras.Sequential([
    # Capa de entrada y primera capa convolucional
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Segunda capa convolucional
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Tercer capa convolucional
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Aplanar la entrada
    tf.keras.layers.Flatten(),
    # Primera capa oculta con 512 neuronas
    tf.keras.layers.Dense(512, activation='relu'),
    # Segunda capa oculta con 512 neuronas
    tf.keras.layers.Dense(256, activation='relu'),
    # Tercer capa oculta con 512 neuronas
    tf.keras.layers.Dense(64, activation='relu'),
    # Capa de salida con 3 neuronas (Hombre, Mujer)
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compilación y entrenamiento del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, epochs=20)

# Guardar el historial de entrenamiento
with open('training_validation_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

# Guardar el modelo entrenado
model.save('gender_validation_model.h5')