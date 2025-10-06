import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import joblib
import os
import numpy as np

# Definindo caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'field_notes_database.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'rnn_model.h5')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer.pkl')
os.makedirs(os.path.join(BASE_DIR, 'model_artifacts'), exist_ok=True) # Cria pasta para artefatos

print(f"Lendo dados de: {DATA_PATH}")

# 1. Carregar e preparar os dados
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Erro: Arquivo de dados não encontrado em {DATA_PATH}. Certifique-se de ter executado a Etapa 2.")
    exit()

# Mapear rótulos de texto para valores numéricos (0 e 1)
label_map = {'rotina': 0, 'urgente': 1}
df['rotulo_encoded'] = df['rotulo'].map(label_map)

X = df['nota'].astype(str).values
y = df['rotulo_encoded'].values

# Divide em treino e teste
X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Pré-processamento de Texto
# Configurações do Tokenizer
vocab_size = 1000  # Tamanho máximo do vocabulário
max_len = 50       # Comprimento máximo da sequência de tokens

# Cria e ajusta o Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

# Converte o texto em sequências de números (tokens)
train_sequences = tokenizer.texts_to_sequences(X_train_text)
test_sequences = tokenizer.texts_to_sequences(X_test_text)

# Preenche sequências para ter o mesmo comprimento (padding)
X_train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
X_test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')


# 3. Construção e Treinamento do Modelo LSTM (RNN)
model = Sequential([
    # Camada de Embedding: Converte tokens em vetores densos
    Embedding(vocab_size, 16, input_length=max_len), 
    
    # Camada LSTM: captura dependências de sequência (memória)
    LSTM(32),
    
    # Camada Densa de Saída: 1 neurônio e 'sigmoid' para classificação binária
    Dense(1, activation='sigmoid') 
])

# Compilação
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Iniciando o treinamento do Modelo LSTM...")

# Treinamento
history = model.fit(X_train_padded, y_train, epochs=20, batch_size=4, verbose=0)

# 4. Avaliação (Opcional, mas útil)
loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
print(f"\nTreinamento concluído.")
print(f"Acurácia nos dados de teste: {accuracy:.4f}")

# 5. Salvar o modelo e o Tokenizer
model.save(os.path.join(BASE_DIR, 'model_artifacts', 'rnn_model.h5'))
joblib.dump(tokenizer, os.path.join(BASE_DIR, 'model_artifacts', 'tokenizer.pkl'))

print(f"Modelo salvo em: {os.path.join(BASE_DIR, 'model_artifacts', 'rnn_model.h5')}")
print(f"Tokenizer salvo em: {os.path.join(BASE_DIR, 'model_artifacts', 'tokenizer.pkl')}")