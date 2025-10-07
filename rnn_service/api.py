import os
import csv
import joblib # Novo: Para carregar o Tokenizer
import numpy as np # Novo: Para manipular arrays
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model # Novo: Para carregar o modelo Keras
from tensorflow.keras.preprocessing.sequence import pad_sequences # Novo: Para padronizar sequências

app = Flask(__name__)
CSV_FILE = 'field_notes_database.csv'
HEADERS = ["nota", "rotulo"]

# Variáveis globais para armazenar o modelo e o pré-processador
rnn_model = None
tokenizer = None
MAX_LEN = 50 # Comprimento máximo da sequência usado no treinamento
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_artifacts')

# ----------------------------------------------------
# Função de Carga (Nova)
# ----------------------------------------------------
def load_rnn_artifacts():
    """Carrega o modelo RNN (LSTM) e o Tokenizer na memória."""
    global rnn_model, tokenizer
    try:
        # Carrega o modelo
        model_path = os.path.join(MODEL_DIR, 'rnn_model.h5')
        rnn_model = load_model(model_path)
        print(f"Modelo RNN carregado com sucesso de: {model_path}")
        
        # Carrega o tokenizer (pré-processador)
        tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.pkl')
        tokenizer = joblib.load(tokenizer_path)
        print(f"Tokenizer carregado com sucesso de: {tokenizer_path}")
        
    except Exception as e:
        print(f"ERRO ao carregar artefatos RNN. Execute Etapa 3: {e}")
        rnn_model = None
        tokenizer = None

# ----------------------------------------------------
# Função utilitária para salvar os dados (Sem Alteração na Lógica)
# ----------------------------------------------------
def append_to_csv(data):
    file_exists = os.path.isfile(CSV_FILE)
    
    try:
        with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=HEADERS, delimiter=',')
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
        
        return True
    except Exception as e:
        print(f"Erro ao salvar CSV: {e}")
        return False

# ----------------------------------------------------
# ENDPOINT DE INGESTÃO (/log) (Sem Alteração na Lógica)
# ----------------------------------------------------
@app.route('/log/note', methods=['POST'])
def log_note():
    """Recebe uma nota de texto e o rótulo (urgente ou rotina)."""
    
    try:
        data = request.get_json()
        if not data or 'nota' not in data or 'rotulo' not in data:
            return jsonify({"status": "error", "message": "Dados JSON inválidos. Requer 'nota' e 'rotulo'."}), 400
    except Exception:
        return jsonify({"status": "error", "message": "Formato JSON inválido."}), 400

    note = data['nota']
    label = data['rotulo'].lower()
    
    if label not in ['urgente', 'rotina']:
        return jsonify({"status": "error", "message": "Rótulo inválido. Use 'urgente' ou 'rotina'."}), 400

    save_data = {"nota": note, "rotulo": label}

    if append_to_csv(save_data):
        return jsonify({"status": "success", "message": "Nota de campo registrada com sucesso"}), 201
    else:
        return jsonify({"status": "error", "message": "Falha ao salvar a nota no sistema de arquivos"}), 500

# ----------------------------------------------------
# ENDPOINT DE INFERÊNCIA (/predict) (Novo)
# POST /predict/note
# ----------------------------------------------------
@app.route('/predict/note', methods=['POST'])
def predict_note():
    """Recebe uma nota de texto e retorna a predição de urgência."""
    global rnn_model, tokenizer
    
    if rnn_model is None or tokenizer is None:
        return jsonify({"status": "error", "message": "Modelo ou pré-processador não carregado. Verifique os logs de inicialização."}), 503

    try:
        data = request.get_json()
        if not data or 'nota' not in data:
            return jsonify({"status": "error", "message": "Dados incompletos. Requer 'nota'."}), 400
    except Exception:
        return jsonify({"status": "error", "message": "Formato JSON inválido."}), 400
    
    try:
        input_note = data['nota']
        
        # 1. Pré-processamento: Tokenizar e Padronizar a sequência
        sequence = tokenizer.texts_to_sequences([input_note])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
        
        # 2. Predição
        prediction_proba = rnn_model.predict(padded_sequence)[0][0]
        
        # 3. Decisão final (limite de 0.5)
        prediction_label = "Urgente" if prediction_proba >= 0.5 else "Rotina"
        
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
    # Roda o servidor na porta 5003
    load_rnn_artifacts() # Novo: Carrega o modelo ao iniciar
    print(f"Serviço RNN rodando em http://127.0.0.1:5003")
    app.run(port=5003, debug=True, use_reloader=False)