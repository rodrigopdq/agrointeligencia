import streamlit as st
import requests
import base64
from io import BytesIO
import pandas as pd
import json

# Configurações de URL
FNN_URL = "http://127.0.0.1:5001"
CNN_URL = "http://127.0.0.1:5002"
RNN_URL = "http://127.0.0.1:5003"

st.set_page_config(layout="wide")
st.title("🌱 Agrointeligência MVP - Plataforma de Predição")

# ---------------------------------------------------------
# UTILS
# ---------------------------------------------------------
def call_api(url, endpoint, data):
    """Função genérica para chamar os endpoints de predição."""
    full_url = f"{url}{endpoint}"
    try:
        response = requests.post(full_url, json=data)
        response.raise_for_status() # Levanta erro para status 4xx/5xx
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Erro de Conexão: O serviço Flask em {url} não está rodando."
    except requests.exceptions.RequestException as e:
        return None, f"Erro da API ({response.status_code}): {response.text}"

# ---------------------------------------------------------
# FNN - Predição de Rendimento
# ---------------------------------------------------------
def fnn_tab():
    st.header("Análise de Solo/Clima (FNN)")
    st.subheader("Previsão de Rendimento Alto")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Insira os Dados de Campo:**")
        temperatura = st.slider("Temperatura (°C)", 15.0, 35.0, 25.0, 0.1)
        umidade = st.slider("Umidade do Ar (%)", 40, 95, 65)
        chuva = st.slider("Chuva (mm)", 0, 200, 100)
        ph = st.slider("PH do Solo", 5.0, 7.5, 6.5, 0.1)

        input_data = {
            "temperatura": temperatura,
            "umidade": umidade,
            "chuva": chuva,
            "ph": ph
        }
        
        # Display dos dados inseridos
        st.markdown("---")
        st.json(input_data)


    with col2:
        if st.button("PREDIZER RENDIMENTO (FNN)", key="fnn_predict"):
            with st.spinner('Aguardando resposta do modelo...'):
                response_data, error = call_api(FNN_URL, "/predict/soil_data", input_data)
                
                if error:
                    st.error(error)
                elif response_data and response_data.get('status') == 'success':
                    label = response_data['prediction_label']
                    score = response_data['confidence_score']
                    
                    st.subheader("Resultado da Predição:")
                    
                    if label == "Rendimento Alto":
                        st.success(f"Predição: {label}")
                    else:
                        st.warning(f"Predição: {label}")

                    st.metric(label="Confiança", value=f"{score*100:.2f}%")
                    st.write("O sistema prediz que estas condições de solo/clima levarão a um **Rendimento " + label.upper() + "**.")
                else:
                    st.error("Erro desconhecido na predição.")

# ---------------------------------------------------------
# CNN - Predição de Doença Foliar
# ---------------------------------------------------------
def cnn_tab():
    st.header("Análise de Imagem Foliar (CNN)")
    st.subheader("Classificação de Saúde da Planta")
    
    uploaded_file = st.file_uploader("Carregue uma imagem de folha para análise:", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Imagem Carregada.', width=250)
        
        if st.button("PREDIZER DOENÇA (CNN)", key="cnn_predict"):
            with st.spinner('Aguardando resposta do modelo...'):
                # 1. Conversão para Base64
                file_bytes = uploaded_file.getvalue()
                image_base64 = base64.b64encode(file_bytes).decode('utf-8')
                
                input_data = {"image_base64": image_base64}

                # 2. Chamada à API
                response_data, error = call_api(CNN_URL, "/predict/leaf_image", input_data)
                
                if error:
                    st.error(error)
                elif response_data and response_data.get('status') == 'success':
                    label = response_data['prediction_label']
                    score = response_data['confidence_score']

                    st.subheader("Resultado da Predição:")
                    
                    if label == "Saudável":
                        st.success(f"Diagnóstico: {label}")
                        st.balloons()
                    else:
                        st.error(f"Diagnóstico: {label}")

                    st.metric(label="Confiança (Prob. de ser Doente)", value=f"{score*100:.2f}%")
                    
                    if label == "Doente":
                         st.markdown(f"**Atenção:** A imagem apresenta sinais de doença com **{score*100:.2f}%** de confiança.")
                    else:
                         st.markdown("A folha parece estar saudável.")
                else:
                    st.error("Erro desconhecido na predição.")

# ---------------------------------------------------------
# RNN - Classificação de Notas de Campo
# ---------------------------------------------------------
def rnn_tab():
    st.header("Análise de Notas de Campo (RNN/LSTM)")
    st.subheader("Classificação de Urgência")
    
    note_text = st.text_area("Insira a nota de campo do agrônomo/técnico:", 
                             "O solo na área recém-plantada está compactando muito rapidamente. Requer subsolagem urgente.")
    
    input_data = {"nota": note_text}
    st.markdown("---")
    st.json(input_data)
    
    if st.button("PREDIZER URGÊNCIA (RNN)", key="rnn_predict"):
        with st.spinner('Aguardando resposta do modelo...'):
            response_data, error = call_api(RNN_URL, "/predict/note", input_data)
            
            if error:
                st.error(error)
            elif response_data and response_data.get('status') == 'success':
                label = response_data['prediction_label']
                score = response_data['confidence_score']
                
                st.subheader("Resultado da Predição:")
                
                if label == "Urgente":
                    st.error(f"Prioridade: {label}")
                else:
                    st.info(f"Prioridade: {label}")

                st.metric(label="Confiança (Prob. de ser Urgente)", value=f"{score*100:.2f}%")
                
                if label == "Urgente":
                    st.markdown("**Recomendação:** A nota exige atenção imediata.")
                else:
                    st.markdown("A nota pode ser tratada dentro da rotina normal.")
            else:
                st.error("Erro desconhecido na predição.")


# ---------------------------------------------------------
# NAVEGAÇÃO PRINCIPAL
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["FNN - Solo", "CNN - Imagem", "RNN - Texto"])

with tab1:
    fnn_tab()

with tab2:
    cnn_tab()

with tab3:
    rnn_tab()

st.sidebar.title("Instruções de Execução")
st.sidebar.markdown("""
1.  **Iniciar os 3 Serviços Flask:**
    Abra 3 terminais separados e execute:
    - `python fnn_service/api.py` (Porta 5001)
    - `python cnn_service/api.py` (Porta 5002)
    - `python rnn_service/api.py` (Porta 5003)

2.  **Iniciar o Dashboard Streamlit:**
    No quarto terminal (na pasta raiz), execute:
    - `streamlit run app.py`

3.  **Teste o Sistema:**
    Use as abas para enviar dados de solo, imagens e texto para os modelos treinados.
""")