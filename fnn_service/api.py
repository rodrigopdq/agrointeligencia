import os
import csv
import joblib 
import numpy as np 
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model 

# Configuração
app = Flask(__name__)
CSV_FILE = 'soil_database.csv'
HEADERS = ["temperatura", "umidade", "chuva", "ph", "rendimento_alto"] 
FEATURES = ['temperatura', 'umidade', 'chuva', 'ph'] 

# Variáveis globais para armazenar o modelo e o pré-processador
fnn_model = None
scaler = None
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_artifacts')

# ----------------------------------------------------
# Função de Carga (Nova)
# ----------------------------------------------------
def load_fnn_artifacts():
    """Carrega o modelo FNN e o scaler na memória."""
    global fnn_model, scaler
    try:
        # Carrega o modelo
        model_path = os.path.join(MODEL_DIR, 'fnn_model.h5')
        fnn_model = load_model(model_path)
        print(f"Modelo FNN carregado com sucesso de: {model_path}")
        
        # Carrega o scaler (pré-processador)
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        scaler = joblib.load(scaler_path)
        print(f"Scaler carregado com sucesso de: {scaler_path}")
        
    except Exception as e:
        print(f"ERRO ao carregar artefatos FNN. Execute Etapa 3: {e}")
        fnn_model = None
        scaler = None

# ----------------------------------------------------
# Função utilitária para salvar os dados no CSV
# ----------------------------------------------------
def append_to_csv(data):
    file_exists = os.path.isfile(CSV_FILE)
    
    try:
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=HEADERS, delimiter=',')
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
        return True
    except Exception as e:
        print(f"Erro ao salvar CSV: {e}")
        return False

# ----------------------------------------------------
# ENDPOINT DE INGESTÃO (/log)
# ----------------------------------------------------
@app.route('/log/soil_data', methods=['POST'])
def log_soil_data():
    """Recebe um JSON com dados de solo/clima e anexa ao CSV."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Nenhum dado JSON fornecido"}), 400
    except Exception:
        return jsonify({"status": "error", "message": "JSON inválido"}), 400

    required_keys = FEATURES + ['rendimento_alto']
    if not all(key in data for key in required_keys):
         return jsonify({"status": "error", "message": f"Dados incompletos. Requer todos os campos: {required_keys}"}), 400

    if append_to_csv(data):
        return jsonify({"status": "success", "message": "Dados de solo registrados com sucesso"}), 201
    else:
        return jsonify({"status": "error", "message": "Falha ao salvar os dados no sistema de arquivos"}), 500

# ----------------------------------------------------
# ENDPOINT DE INFERÊNCIA (/predict) (NOVO)
# POST /predict/soil_data
# ----------------------------------------------------
@app.route('/predict/soil_data', methods=['POST'])
def predict_soil_data():
    """Recebe novos dados de solo/clima e retorna uma predição de rendimento."""
    global fnn_model, scaler
    
    if fnn_model is None or scaler is None:
        return jsonify({"status": "error", "message": "Modelo ou pré-processador não carregado. Verifique os logs de inicialização."}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Nenhum dado JSON fornecido"}), 400
    except Exception:
        return jsonify({"status": "error", "message": "JSON inválido"}), 400
    
    if not all(key in data for key in FEATURES):
        return jsonify({"status": "error", "message": f"Dados incompletos. Requer os campos: {FEATURES}"}), 400

    try:
        # 1. Preparar a entrada de dados
        input_data = [data[f] for f in FEATURES]
        input_array = np.array(input_data).reshape(1, -1) 

        # 2. Pré-processamento
        input_scaled = scaler.transform(input_array)
        
        # 3. Predição
        prediction_proba = fnn_model.predict(input_scaled)[0][0]
        
        # 4. Decisão final (limite de 0.5)
        prediction_label = "Rendimento Alto" if prediction_proba >= 0.5 else "Rendimento Normal/Baixo"
        
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
    # Roda o servidor na porta 5001
    load_fnn_artifacts() 
    print(f"Serviço FNN rodando em http://127.0.0.1:5001")
    # use_reloader=False é importante para evitar que o modelo carregue duas vezes
    app.run(port=5001, debug=True, use_reloader=False)