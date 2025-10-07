import os
import time
import base64
import numpy as np # Novo: Para manipular arrays
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model # Novo: Para carregar o modelo Keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img # Novo: Para processar a imagem

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads' 

# Variáveis globais para armazenar o modelo
cnn_model = None
IMG_SIZE = (64, 64) # Tamanho que o modelo foi treinado
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_artifacts')

# ----------------------------------------------------
# Função de Carga (Nova)
# ----------------------------------------------------
def load_cnn_artifact():
    """Carrega o modelo CNN na memória."""
    global cnn_model
    try:
        model_path = os.path.join(MODEL_DIR, 'cnn_model.h5')
        cnn_model = load_model(model_path)
        print(f"Modelo CNN carregado com sucesso de: {model_path}")
        
    except Exception as e:
        print(f"ERRO ao carregar artefatos CNN. Execute Etapa 3: {e}")
        cnn_model = None

# ----------------------------------------------------
# ENDPOINT DE INGESTÃO (/log) (Sem Alteração na Lógica)
# ----------------------------------------------------
@app.route('/log/leaf_image', methods=['POST'])
def log_leaf_image():
    """Recebe imagem em Base64 e rótulo, decodifica e salva."""
    
    try:
        data = request.get_json()
        
        if not data or 'image_base64' not in data or 'label' not in data:
            return jsonify({"status": "error", "message": "JSON inválido. Requer 'image_base64' e 'label'."}), 400

        image_base64 = data['image_base64']
        label = data['label'].lower()
        
        if label not in ['saudavel', 'doente']:
            return jsonify({"status": "error", "message": "Rótulo inválido. Use 'saudavel' ou 'doente'."}), 400

        target_dir = os.path.join(UPLOAD_FOLDER, label)
        os.makedirs(target_dir, exist_ok=True) 

        try:
            image_data = base64.b64decode(image_base64)
        except Exception:
            return jsonify({"status": "error", "message": "Falha na decodificação do Base64."}), 400
        
        filename = f"image_{int(time.time())}_{label}.jpg"
        file_path = os.path.join(target_dir, filename)

        with open(file_path, 'wb') as f:
            f.write(image_data)

        return jsonify({
            "status": "success", 
            "message": f"Imagem registrada em: {file_path}",
            "filename": filename
        }), 201

    except Exception as e:
        return jsonify({"status": "error", "message": f"Erro interno do servidor: {e}"}), 500

# ----------------------------------------------------
# ENDPOINT DE INFERÊNCIA (/predict) (Novo)
# POST /predict/leaf_image
# ----------------------------------------------------
@app.route('/predict/leaf_image', methods=['POST'])
def predict_leaf_image():
    """Recebe uma imagem em Base64 e retorna a predição de doença."""
    global cnn_model
    
    if cnn_model is None:
        return jsonify({"status": "error", "message": "Modelo CNN não carregado. Verifique os logs de inicialização."}), 503

    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({"status": "error", "message": "Dados incompletos. Requer 'image_base64'."}), 400

        image_base64 = data['image_base64']
        
        # 1. Decodificar Base64 e preparar para Keras
        image_data = base64.b64decode(image_base64)
        
        # Cria um objeto temporário em memória para Keras processar
        from io import BytesIO
        image_stream = BytesIO(image_data)
        
        # Carrega, redimensiona e converte para array
        img = load_img(image_stream, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        
        # Adiciona uma dimensão de batch (1, 64, 64, 3)
        img_array = np.expand_dims(img_array, axis=0) 
        
        # Normaliza (dividir por 255)
        img_array /= 255.0
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Falha no pré-processamento da imagem: {e}"}), 400

    try:
        # 2. Predição
        prediction_proba = cnn_model.predict(img_array)[0][0]
        
        # 3. Decisão final (0 = Saudável, 1 = Doente)
        prediction_label = "Doente" if prediction_proba >= 0.5 else "Saudável"
        
        return jsonify({
            "status": "success",
            "prediction_label": prediction_label,
            "confidence_score": float(prediction_proba)
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": f"Erro interno durante a predição: {e}"}), 500

# ----------------------------------------------------
# Execução do Servidor
# ----------------------------------------------------
if __name__ == '__main__':
    # Roda o servidor na porta 5002
    load_cnn_artifact() # Novo: Carrega o modelo ao iniciar
    print(f"Serviço CNN rodando em http://127.0.0.1:5002")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(port=5002, debug=True, use_reloader=False)