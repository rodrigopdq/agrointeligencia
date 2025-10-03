import os
import time
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)
# Pasta raiz onde as imagens serão salvas
UPLOAD_FOLDER = 'uploads' 

# ----------------------------------------------------
# ENDPOINT DE INGESTÃO (/log)
# ----------------------------------------------------
@app.route('/log/leaf_image', methods=['POST'])
def log_leaf_image():
    """Recebe imagem em Base64 e rótulo, decodifica e salva."""
    
    try:
        data = request.get_json()
        
        # 1. Validação da entrada
        if not data or 'image_base64' not in data or 'label' not in data:
            return jsonify({"status": "error", "message": "JSON inválido. Requer 'image_base64' e 'label'."}), 400

        image_base64 = data['image_base64']
        label = data['label'].lower()
        
        # 2. Validação do rótulo
        if label not in ['saudavel', 'doente']:
            return jsonify({"status": "error", "message": "Rótulo inválido. Use 'saudavel' ou 'doente'."}), 400

        # 3. Definição do caminho de salvamento
        # Ex: cnn_service/uploads/saudavel
        target_dir = os.path.join(UPLOAD_FOLDER, label)
        
        # Cria a pasta se ela não existir (muito importante!)
        os.makedirs(target_dir, exist_ok=True) 

        # 4. Decodificação e nomeação do arquivo
        try:
            # Decodifica o Base64 para os bytes da imagem
            image_data = base64.b64decode(image_base64)
        except Exception:
            return jsonify({"status": "error", "message": "Falha na decodificação do Base64."}), 400
        
        # Cria um nome de arquivo único baseado no timestamp atual
        filename = f"image_{int(time.time())}_{label}.jpg"
        file_path = os.path.join(target_dir, filename)

        # 5. Salvamento da imagem
        with open(file_path, 'wb') as f:
            f.write(image_data)

        return jsonify({
            "status": "success", 
            "message": f"Imagem registrada em: {file_path}",
            "filename": filename
        }), 201

    except Exception as e:
        # Erro genérico do servidor
        return jsonify({"status": "error", "message": f"Erro interno do servidor: {e}"}), 500

# ----------------------------------------------------
# Execução do Servidor
# ----------------------------------------------------
if __name__ == '__main__':
    # Roda o servidor na porta 5002
    print(f"Serviço CNN rodando em http://127.0.0.1:5002")
    # Certifique-se de que a pasta 'uploads' exista
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(port=5002, debug=True)