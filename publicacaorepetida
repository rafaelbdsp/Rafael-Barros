import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re
from datetime import datetime
from io import BytesIO

# Função para extrair data do nome do arquivo
def extrair_data(nome_arquivo):
    match = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", nome_arquivo)
    if match:
        return datetime.strptime(match.group(0), "%d.%m.%Y")
    return None

# Configurações iniciais
st.set_page_config(page_title="Analisador Semântico de Publicações", page_icon="🔍", layout="centered")

st.title("🔍 Analisador Semântico de Publicações Jurídicas")
st.write("Envie os arquivos .xlsx das publicações, ajuste o limiar e clique em **Iniciar Comparação**.")

# Ajuste de limiar
limiar = st.slider("Limiar de Similaridade", 0.5, 1.0, 0.85, 0.01)

# Upload de arquivos
uploaded_files = st.file_uploader("📂 Envie as planilhas .xlsx", type="xlsx", accept_multiple_files=True)

# Filtro para mostrar apenas Duplicatas Exatas
mostrar_exatas = st.checkbox("Mostrar apenas Duplicatas Exatas no resultado", value=False)

if uploaded_files and st.button("🚀 Iniciar Comparação"):
    st.info("⚙️ Carregando modelo semântico... Isso pode levar alguns segundos.")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    dfs = {}
    for file in uploaded_files:
        df = pd.read_excel(file)
        df.columns = df.columns.str.strip()
        df = df[["Processo", "Intimação"]].dropna()
        dfs[file.name] = df
        st.success(f"{file.name}: {len(df)} publicações carregadas.")

    resultados = []
    total_comparacoes = sum(len(df1) * len(df2) for i, df1 in enumerate(dfs.values()) for j, df2 in enumerate(dfs.values()) if i < j)
    progresso = st.progress(0)
    passo = 0

    for i, (nome_i, df_i) in enumerate(dfs.items()):
        for j, (nome_j, df_j) in enumerate(dfs.items()):
            if i >= j:
                continue

            st.write(f"🔎 Comparando: {nome_i} com {nome_j}...")

            for _, row_i in df_i.iterrows():
                processo = str(row_i["Processo"]).strip()
                texto_i = str(row_i["Intimação"]).strip()

                candidatos = df_j[df_j["Processo"] == processo]

                for _, row_j in candidatos.iterrows():
                    texto_j = str(row_j["Intimação"]).strip()

                    emb = model.encode([texto_i, texto_j], convert_to_tensor=True)
                    score = util.pytorch_cos_sim(emb[0], emb[1]).item()

                    if score >= limiar:
                        motivo = ("Duplicata Exata" if score >= 0.99 else 
                                  "Muito semelhante" if score >= 0.95 else
                                  "Moderadamente semelhante")

                        data_i = extrair_data(nome_i)
                        data_j = extrair_data(nome_j)

                        if data_i and data_j:
                            if data_i > data_j:
                                excluir = nome_i
                                data_antiga = data_j.strftime("%d/%m/%Y")
                            else:
                                excluir = nome_j
                                data_antiga = data_i.strftime("%d/%m/%Y")
                            resultado = (f"Publicação lançada anteriormente no dia {data_antiga}; "
                                         f"Excluir arquivo mais recente: {excluir} - Motivo: {motivo}")
                        else:
                            resultado = f"Revisar - Não foi possível determinar a mais recente - Motivo: {motivo}"

                        resultados.append({
                            "Processo": processo,
                            "Similaridade": round(score, 4),
                            "Publicação 1": texto_i,
                            "Publicação 2": texto_j,
                            "Arquivo 1": nome_i,
                            "Arquivo 2": nome_j,
                            "Resultado": resultado,
                            "Motivo": motivo
                        })

                    passo += 1
                    progresso.progress(min(passo / total_comparacoes, 1.0))

    df_resultado = pd.DataFrame(resultados)

    if mostrar_exatas:
        df_resultado = df_resultado[df_resultado["Motivo"] == "Duplicata Exata"]

    output = BytesIO()
    df_resultado.to_excel(output, index=False)

    st.success(f"✅ Comparação finalizada! {len(df_resultado)} possíveis duplicatas encontradas.")
    st.download_button("📥 Baixar Resultado", output.getvalue(), file_name="comparacao_semantica.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
