import os
import csv
from flask import Flask, request, jsonify

app = Flask(__name__)
CSV_FILE = 'field_notes_database.csv'
HEADERS = ["nota", "rotulo"]

# Função utilitária para salvar os dados
def append_to_csv(data):
    # Verifica se o arquivo existe para saber se precisamos escrever o cabeçalho
    file_exists = os.path.isfile(CSV_FILE)
    
    try:
        # Abrimos o arquivo no modo 'a' (append/anexar)
        with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
            # Usamos DictWriter para escrever o dicionário diretamente no CSV
            writer = csv.DictWriter(file, fieldnames=HEADERS, delimiter=',')

            # Se o arquivo não existe, escrevemos o cabeçalho
            if not file_exists:
                writer.writeheader()
            
            # Escrevemos a linha de dados
            writer.writerow(data)
        
        return True
    except Exception as e:
        print(f"Erro ao salvar CSV: {e}")
        return False

# ----------------------------------------------------
# ENDPOINT DE INGESTÃO (/log)
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
    
    # Validação do rótulo
    if label not in ['urgente', 'rotina']:
        return jsonify({"status": "error", "message": "Rótulo inválido. Use 'urgente' ou 'rotina'."}), 400

    # Dicionário final para salvar
    save_data = {"nota": note, "rotulo": label}

    # Salvar o dado no arquivo CSV
    if append_to_csv(save_data):
        return jsonify({"status": "success", "message": "Nota de campo registrada com sucesso"}), 201
    else:
        return jsonify({"status": "error", "message": "Falha ao salvar a nota no sistema de arquivos"}), 500

# ----------------------------------------------------
# Execução do Servidor
# ----------------------------------------------------
if __name__ == '__main__':
    # Roda o servidor na porta 5003
    print(f"Serviço RNN rodando em http://127.0.0.1:5003")
    app.run(port=5003, debug=True)