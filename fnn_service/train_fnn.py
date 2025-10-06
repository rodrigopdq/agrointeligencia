import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os
import numpy as np

# Definindo caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'soil_database.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'fnn_model.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
os.makedirs(os.path.join(BASE_DIR, 'model_artifacts'), exist_ok=True) # Cria pasta para artefatos

print(f"Lendo dados de: {DATA_PATH}")

# 1. Carregar e preparar os dados
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Erro: Arquivo de dados não encontrado em {DATA_PATH}. Certifique-se de ter executado a Etapa 2.")
    exit()

# Definir Features (X) e Target (y)
features = ['temperatura', 'umidade', 'chuva', 'ph']
target = 'rendimento_alto'

X = df[features].values
y = df[target].values

# 2. Pré-processamento
# Normaliza os dados de entrada
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Divide em treino e teste (necessário para avaliação, mesmo com dataset pequeno)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 3. Construção e Treinamento do Modelo FNN
model = Sequential([
    # Input layer com 4 neurônios (número de features)
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)), 
    Dense(8, activation='relu'),
    # Output layer: 1 neurônio e 'sigmoid' para classificação binária
    Dense(1, activation='sigmoid') 
])

# Compilação
model.compile(optimizer='adam',
              loss='binary_crossentropy', # Perda para classificação binária
              metrics=['accuracy'])

print("Iniciando o treinamento do Modelo FNN...")

# Treinamento
history = model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=0)

# 4. Avaliação (Opcional, mas útil)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTreinamento concluído.")
print(f"Acurácia nos dados de teste: {accuracy:.4f}")

# 5. Salvar o modelo e o pré-processador
model.save(os.path.join(BASE_DIR, 'model_artifacts', 'fnn_model.h5'))
joblib.dump(scaler, os.path.join(BASE_DIR, 'model_artifacts', 'scaler.pkl'))

print(f"Modelo salvo em: {os.path.join(BASE_DIR, 'model_artifacts', 'fnn_model.h5')}")
print(f"Scaler salvo em: {os.path.join(BASE_DIR, 'model_artifacts', 'scaler.pkl')}")