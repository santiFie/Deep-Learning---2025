from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score, classification_report


class Modelo:
    def __init__(self, X_train, y_train, ocultas=[10], funH='relu', alpha=0.001, epocas=200, batch_size=32, solver='adam', random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.ocultas = ocultas
        self.funH = funH
        self.alpha = alpha
        self.epocas = epocas
        self.batch_size = batch_size
        self.solver = solver
        self.random_state = random_state
        self.modelo = None
        self.crear_modelo()

    def crear_modelo(self):
        self.modelo = MLPClassifier(
            hidden_layer_sizes=self.ocultas,
            activation=self.funH,
            alpha=self.alpha,
            max_iter=self.epocas,
            batch_size=self.batch_size,
            solver=self.solver,
            random_state=self.random_state
        )

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.crear_modelo()

    def entrenar(self):
        if self.modelo is None:
            self.crear_modelo()
        self.modelo.fit(self.X_train, self.y_train)

    def predecir(self, X):
        if self.modelo is None:
            raise Exception("El modelo no está entrenado.")
        return self.modelo.predict(X)

    def score(self, X, y):
        if self.modelo is None:
            raise Exception("El modelo no está entrenado.")
        return self.modelo.score(X, y)
    

    def evaluar_modelo_multiple_veces(self, X, y, n_ejecuciones=10):
        """Evaluar modelo múltiples veces para validación estadística"""
        
        accuracies_train = []
        accuracies_test = []
        
        for i in range(n_ejecuciones):
            print(f"\n--- Ejecución {i+1}/{n_ejecuciones} ---")
            
            # División aleatoria DIFERENTE en cada ejecución
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, 
                random_state=i,
                shuffle=True
            )
            
            # Normalizar
            scaler = preprocessing.StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # binarizer = preprocessing.LabelBinarizer()
            # y_train = binarizer.fit_transform(y_train)
            # y_test = binarizer.transform(y_test)

            
            # Crear modelo con semilla diferente
            self.set_params(random_state=i)
            self.crear_modelo()

            # Entrenar
            self.entrenar()
            
            # Evaluar
            acc_train = self.score(X_train, y_train)
            acc_test = self .score(X_test, y_test)
            
            accuracies_train.append(acc_train)
            accuracies_test.append(acc_test)
            
            print(f"Train Accuracy: {acc_train:.4f}")
            print(f"Test Accuracy: {acc_test:.4f}")
        
        return accuracies_train, accuracies_test
