import requests
import base64
import os
import time

# URL do endpoint de ingestão da CNN
CNN_URL = 'http://127.0.0.1:5002/log/leaf_image'

# ----------------------------------------------------
# 1. Função para converter imagem em Base64
# ----------------------------------------------------
def image_to_base64(file_path):
    """Lê um arquivo de imagem e retorna seu conteúdo em string Base64."""
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# ----------------------------------------------------
# 2. Função para enviar a imagem
# ----------------------------------------------------
def send_image_to_cnn(image_path, label):
    """Converte e envia a imagem para a API."""
    base64_data = image_to_base64(image_path)

    payload = {
        "image_base64": base64_data,
        "label": label
    }

    try:
        response = requests.post(CNN_URL, json=payload)
        response.raise_for_status()
        
        result = response.json()
        print(f"   -> SUCESSO ({label}). Mensagem: {result.get('message')}")
        
    except requests.exceptions.RequestException as e:
        print(f"   -> ERRO ao enviar a imagem. Verifique se o serviço da CNN está rodando na porta 5002.")
        print(f"   Detalhe do Erro: {e}")

# ----------------------------------------------------
# 3. Execução Principal
# ----------------------------------------------------
if __name__ == '__main__':
    SOURCE_DIR = r'C:/Users/Usuário/Desktop/Projeto agro/agrointeligencia/cnn_service/Imagens_Para_Enviar'
    
    if not os.path.isdir(SOURCE_DIR):
        print(f"ERRO: O diretório de origem '{SOURCE_DIR}' não foi encontrado.")
        exit()

    print(f"Iniciando ingestão de imagens a partir de: {SOURCE_DIR}")
    
    ingested_count = 0
    
    # Percorre todos os arquivos na pasta
    for filename in os.listdir(SOURCE_DIR):
        file_path = os.path.join(SOURCE_DIR, filename)
        
        # Ignora subpastas
        if os.path.isdir(file_path):
            continue
            
        # 1. Determina o rótulo baseado no nome do arquivo
        label = None
        if '_saudavel' in filename.lower():
            label = 'saudavel'
        elif '_doente' in filename.lower():
            label = 'doente'
        
        if label:
            print(f"Processando arquivo: {filename}")
            send_image_to_cnn(file_path, label)
            ingested_count += 1
            time.sleep(0.1)
        else:
            print(f"AVISO: Arquivo '{filename}' ignorado. Não possui '_saudavel' ou '_doente' no nome.")

    print(f"\n--- Ingestão de imagens concluída. Total de {ingested_count} imagens enviadas. ---")