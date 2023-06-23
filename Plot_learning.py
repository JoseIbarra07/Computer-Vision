import pickle
import matplotlib.pyplot as plt

def plot_learning_curves(history):
    loss = history['loss']
    accuracy = history['accuracy']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 5))

    # Gráfico de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'r', label='Pérdida de entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    # Gráfico de precisión
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'b', label='Precisión de entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    plt.show()

# Cargar el historial de entrenamiento
with open('training_history.pkl', 'rb') as file:
    training_history = pickle.load(file)

# Mostrar las curvas de aprendizaje
plot_learning_curves(training_history)
