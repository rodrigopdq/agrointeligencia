import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Definindo caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'cnn_model.h5')
os.makedirs(os.path.join(BASE_DIR, 'model_artifacts'), exist_ok=True) # Cria pasta para artefatos

# Configurações
IMG_SIZE = (64, 64) # Tamanho da imagem para o modelo
BATCH_SIZE = 4
EPOCHS = 20

# 1. Preparação dos dados
# Removemos o validation_split para usar todos os dados disponíveis para treino.
datagen = ImageDataGenerator(
    rescale=1./255,
)

# Gerador para dados de treino (agora usa todos os dados disponíveis)
train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    seed=42
)

print(f"\nTotal de imagens para treino: {train_generator.samples}")

# 2. Construção e Treinamento do Modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid') 
])

# Compilação
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\nIniciando o treinamento do Modelo CNN...")

# Treinamento
# Ajustamos steps_per_epoch para garantir que o treinamento possa rodar mesmo com poucos dados.
if train_generator.samples > 0:
    steps = train_generator.samples // BATCH_SIZE
else:
    steps = 1 # Garante que o treinamento rode pelo menos uma vez

model.fit(
    train_generator,
    steps_per_epoch=steps,
    epochs=EPOCHS,
    verbose=0
)

# 3. Conclusão
print(f"\nTreinamento concluído. O modelo foi treinado com {train_generator.samples} amostras.")

# 4. Salvar o modelo
model.save(os.path.join(BASE_DIR, 'model_artifacts', 'cnn_model.h5'))
print(f"Modelo salvo em: {os.path.join(BASE_DIR, 'model_artifacts', 'cnn_model.h5')}")