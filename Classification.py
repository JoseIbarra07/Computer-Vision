import cv2
import numpy as np
import tensorflow as tf

def capture_image(image_path):
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            cv2.imshow('Video en tiempo real', frame)

            # Tomar la foto en la primera iteración y salir del bucle
            cv2.imwrite(image_path, frame)
            break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def classify_image(image_path):
    # Cargar el modelo entrenado
    model = tf.keras.models.load_model('gender_classification_model.h5')

    # Leer y preprocesar la imagen
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Clasificar la imagen
    predictions = model.predict(image)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index] * 100

    # Mostrar la clasificación y la confianza
    classes = ['Hombre', 'Mujer']
    print(f"Clasificación: {classes[class_index]}")
    print(f"Confianza: {confidence:.2f}%")

# Uso de las funciones capture_image y classify_image
if __name__ == "__main__":
    image_path = "captured_image.jpg"
    capture_image(image_path)
    classify_image(image_path)
