import cv2
import numpy as np
import tensorflow as tf

def capture_and_classify(image_path):
    # Cargar el modelo entrenado
    model = tf.keras.models.load_model('gender_classification_model.h5')
    classes = ['Hombre', 'Mujer']

    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()

            # Preprocesar la imagen
            image = cv2.resize(frame, (100, 100))
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

            # Clasificar la imagen
            predictions = model.predict(image)
            class_index = np.argmax(predictions)
            confidence = predictions[0][class_index] * 100

            # Mostrar la clasificación y la confianza en el cuadro
            label = f"{classes[class_index]}: {confidence:.2f}%"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Mostrar el cuadro con la clasificación
            cv2.imshow('Video en tiempo real', frame)

            # Salir con la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_classify("captured_image.jpg")
