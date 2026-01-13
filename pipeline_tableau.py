import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import callbacks

# Configuración de semillas para reproducibilidad
np.random.seed(5)
tf.random.set_seed(5)

def run_pipeline():

    # 1. Ingesta y Preprocesamiento de Datos
    df = pd.read_csv('/home/santi/Documentos/Cuarto/Deep Learning/repositorio/Datos/creditcard.csv')
    
    # Escalado de Amount para que el modelo no se sesgue por montos altos
    df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # Eliminamos Time
    df = df.drop(['Time'], axis=1)

    # División Train/Test (80/20)
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

    # Guardamos las etiquetas del Test para el CSV final
    Y_test = X_test['Class']
    
    # Preparamos el Train: Solo datos normales (Clase 0)
    X_train = X_train[X_train.Class == 0]
    X_train = X_train.drop(['Class'], axis=1)
    
    # Preparamos el Test: Eliminamos la clase para entrar al modelo
    X_test_input = X_test.drop(['Class'], axis=1)

    X_train = X_train.values
    X_test_input = X_test_input.values

    # 2. Construcción y Entrenamiento del Modelo
    dim_entrada = X_train.shape[1]  # 29 columnas
    
    capa_entrada = Input(shape=(dim_entrada,))
    encoder = Dense(20, activation='tanh')(capa_entrada)
    encoder = Dense(14, activation='relu')(encoder) 
    # Espacio latente
    decoder = Dense(20, activation='tanh')(encoder)
    decoder = Dense(dim_entrada, activation='relu')(decoder)

    autoencoder = Model(inputs=capa_entrada, outputs=decoder)
    
    # Optimizador SGD. Se podría haber usado Adam u otro.
    sgd = SGD(learning_rate=0.01)
    autoencoder.compile(optimizer=sgd, loss='mse')

    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )

    # Entrenamiento
    autoencoder.fit(
        X_train, X_train,
        epochs=50, # Reducido para demo, puedes subirlo a 100
        batch_size=32,
        shuffle=True,
        validation_data=(X_test_input, X_test_input),
        verbose=1,
        callbacks=[early_stopping]
    )

    # 3. Generación de Resultados
    
    # Predicción (Reconstrucción)
    X_pred = autoencoder.predict(X_test_input)
    
    # Cálculo del Error Cuadrático Medio (ECM) por transacción
    ecm = np.mean(np.power(X_test_input - X_pred, 2), axis=1)

    # --- EXPORTACIÓN 1: DATOS TRANSACCIONALES (Para Box Plot) ---
    df_resultados = pd.DataFrame({
        'Reconstruction_Error': ecm,
        'Class_Real': Y_test.values,  # 0 = Normal, 1 = Fraude
        'Amount_Scaled': X_test['Amount'].values # Opcional: Para contexto visual
    })

    # Etiquetamos mayor legibilidad en Tableau
    df_resultados['Etiqueta'] = df_resultados['Class_Real'].map({0: 'Normal', 1: 'Fraude'})
    
    df_resultados.to_csv('tableau_resultados_test.csv', index=False)

    # --- EXPORTACIÓN 2: CURVA PRECISION-RECALL (Para Gráfico de Líneas) ---
    precision, recall, umbrales = precision_recall_curve(Y_test, ecm)
    
    # Creamos un DataFrame con los puntos de la curva
    # Nota: precision y recall tienen un elemento más que umbrales, ajustamos longitud
    df_curva = pd.DataFrame({
        'Umbral': np.append(umbrales, np.nan), # Agregamos NaN al final para igualar largos
        'Precision': precision,
        'Recall': recall
    })
    
    df_curva.to_csv('tableau_curva_pr.csv', index=False)
    
    print("Archivos generados:")
    print("   1. tableau_resultados_test.csv (Usar para Box Plot)")
    print("   2. tableau_curva_pr.csv (Usar para Línea Precision-Recall)")

if __name__ == "__main__":
    run_pipeline()