import os
import streamlit as st
from pydantic_ai import Agent
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# Configuração da API Key (substitua pelo seu método de armazenamento seguro)
os.environ["OPENAI_API_KEY"] = "API_KEY"

# Interface do Streamlit
st.set_page_config(page_title="Gerador de Questões - Engenharia de Software", page_icon="📘", layout="wide")
st.title("📘 Gerador de Questões sobre Engenharia de Software")
st.write("Faça upload de um PDF e obtenha questões de múltipla escolha baseadas no conteúdo!")

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
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(texts, OpenAIEmbeddings())

    # Entrada do usuário para personalização
    user_question = st.text_area("✏️ Escreva um complemento para sua pergunta (opcional)", "")
    if st.button("🔍 Gerar Perguntas"):
        # Buscar contexto relevante
        docs_relevantes = vectorstore.similarity_search(user_question, k=3)
        contexto = "\n".join([doc.page_content for doc in docs_relevantes])


        # Modelo de resposta
        class Resposta(BaseModel):
            resposta: str

            def __repr__(self):
                return f"Resposta({self.resposta})"


        # Criar o agente
        agente_rag = Agent(
            model="gpt-4o-mini",
            result_type=Resposta,
            system_prompt="Você é um agente auxiliar de um professor especializado em engenharia de software."
        )

        # Gerar resposta
        response = agente_rag.run_sync(
            user_prompt=f"Com base nestas informações:\n{contexto}\n\nGere 5 perguntas de múltipla escolha. Aumente o nível de dificuldade a cada pergunta."
        )

        # Exibir as perguntas
        st.subheader("📌 Perguntas Geradas:")
        st.markdown(f"> {response.data.resposta}")

        # Excluir o arquivo temporário
        os.remove(file_path)
