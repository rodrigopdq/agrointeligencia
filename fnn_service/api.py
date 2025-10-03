import os
import csv
from flask import Flask, request, jsonify

# Configuração
app = Flask(__name__)
CSV_FILE = 'soil_database.csv'
# Definimos o cabeçalho esperado, incluindo a coluna alvo 'rendimento_alto'
HEADERS = ["temperatura", "umidade", "chuva", "ph", "rendimento_alto"] 

# ----------------------------------------------------
# Função utilitária para salvar os dados no CSV
# ----------------------------------------------------
def append_to_csv(data):
    # Verifica se o arquivo já existe. Isso é para saber se precisamos escrever o cabeçalho.
    file_exists = os.path.isfile(CSV_FILE)
    
    try:
        # Abrimos o arquivo no modo 'a' (append/anexar). newline='' evita linhas em branco.
        with open(CSV_FILE, mode='a', newline='') as file:
            # Usamos DictWriter para escrever os dados do dicionário (JSON) diretamente no CSV
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
# POST /log/soil_data
# ----------------------------------------------------
@app.route('/log/soil_data', methods=['POST'])
def log_soil_data():
    """Recebe um JSON com dados de solo/clima e anexa ao CSV."""
    
    # 1. Obter e validar o JSON
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Nenhum dado JSON fornecido"}), 400
    except Exception:
        return jsonify({"status": "error", "message": "JSON inválido"}), 400

    # 2. Validação dos campos obrigatórios
    # Garante que todos os campos do HEADERS estão presentes no JSON
    if not all(key in data for key in HEADERS):
        return jsonify({"status": "error", "message": f"Dados incompletos. Requer todos os campos: {HEADERS}"}), 400

    # 3. Salvar o dado no arquivo CSV
    if append_to_csv(data):
        return jsonify({"status": "success", "message": "Dados de solo registrados com sucesso"}), 201
    else:
        return jsonify({"status": "error", "message": "Falha ao salvar os dados no sistema de arquivos"}), 500

# ----------------------------------------------------
# Execução do Servidor
# ----------------------------------------------------
if __name__ == '__main__':
    # Roda o servidor na porta 5001
    print(f"Serviço FNN rodando em http://127.0.0.1:5001")
    app.run(port=5001, debug=True)