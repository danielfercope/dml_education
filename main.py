import os
from pydantic_ai import Agent
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# 🔑 Configurar a API Key da OpenAI
os.environ["OPENAI_API_KEY"] = "API_KEY"

# 📝 Base de conhecimento (exemplo: textos salvos em um arquivo)
loader = PyPDFLoader("2004_tecnologia_informacao_pbqp.pdf")
documents = loader.load()


#teste
# 🔍 Criar embeddings e armazenar vetores
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
texts = text_splitter.split_documents(documents)
vectorstore = FAISS.from_documents(texts, OpenAIEmbeddings())

# 🎯 Definir o modelo de resposta
class Resposta(BaseModel):
    resposta: str

    def __repr__(self):
        return f"Resposta({self.resposta})"

# 📌 Criar o agente
agente_rag = Agent(
    model="gpt-4o-mini",
    result_type=Resposta,
    system_prompt= "Você é um agente auxiliar de um professor especializado em engenharia de software."

)

# 🏷️ Pergunta do usuário
pergunta = input("Qual sua dúvida? ")

# 🔎 Buscar contexto relevante
docs_relevantes = vectorstore.similarity_search(pergunta, k=3)
contexto = "\n".join([doc.page_content for doc in docs_relevantes])
print(contexto)

# 📢 Gerar resposta usando RAG
response = agente_rag.run_sync(
    user_prompt=f"Com base nestas informações:\n{contexto}\n\nGere 5 perguntas de multipla escolha. Aumente o nivel de dificuldade a cada pergunta."
)

print(response)
