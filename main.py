import os
import streamlit as st
from pydantic_ai import Agent
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

os.environ["OPENAI_API_KEY"] = "API_KEY"

# Interface do Streamlit
st.set_page_config(page_title="Gerador de Questões - Engenharia de Software", page_icon="📘", layout="wide")
st.title("📘 Gerador de Questões sobre Engenharia de Software")
st.write("Faça upload de um PDF e personalize a geração de questões!")

# Upload do arquivo PDF
uploaded_file = st.file_uploader("📂 Faça upload do seu arquivo PDF", type=["pdf"])

if uploaded_file is not None:
    # Salvar temporariamente o arquivo
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("📄 Arquivo carregado com sucesso!")

    # Carregar e processar o PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(texts, OpenAIEmbeddings())

    # Personalização pelo professor
    num_questions = st.slider("🔢 Quantidade de perguntas", min_value=1, max_value=10, value=5)
    difficulty = st.selectbox("🎯 Nível de dificuldade", ["Fácil", "Médio", "Difícil"], index=1)

    if st.button("🚀 Gerar História e Questões"):
        # Buscar contexto relevante
        docs_relevantes = vectorstore.similarity_search("", k=5)
        contexto = "\n".join([doc.page_content for doc in docs_relevantes])


        # Modelo de resposta
        class Resposta(BaseModel):
            historia: str
            questoes: list


        # Criar o agente
        agente_rag = Agent(
            model="gpt-4o-mini",
            result_type=Resposta,
            system_prompt="Você é um agente especializado em engenharia de software que gera histórias e questões baseadas em documentos fornecidos."
        )

        # Gerar história e questões
        response = agente_rag.run_sync(
            user_prompt=f"Baseando-se no seguinte conteúdo:\n{contexto}\n\nCrie uma história completa que introduza o tema do documento. Depois, gere {num_questions} perguntas de múltipla escolha com nível de dificuldade {difficulty}. Forneça o gabarito para cada questão."
        )

        # Edição pelo professor
        historia_editada = st.text_area("📝 História Gerada:", value=response.data.historia, height=200)
        questoes_formatadas = "\n\n".join(
            [
                f"🔹 {q['pergunta']}\n"
                + "\n".join([f"   - {alt}" for alt in q['opcoes']])
                + f"\n✅ Resposta correta: {q['gabarito']}"
                for q in response.data.questoes
            ]
        )

        questoes_editadas = st.text_area("📌 Questões Geradas:", value=questoes_formatadas, height=300)

        if st.button("✅ Salvar Questões e História"):
            st.success("História e questões salvas com sucesso!")

    # Excluir o arquivo temporário
    os.remove(file_path)
