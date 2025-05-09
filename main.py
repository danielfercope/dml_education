import os
import re
from pydantic_ai import Agent
from pydantic import BaseModel
from google.oauth2 import service_account
from googleapiclient.discovery import build

SERVICE_ACCOUNT_FILE = ''
SCOPES = ['https://www.googleapis.com/auth/forms.body']

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('forms', 'v1', credentials=credentials)

os.environ["OPENAI_API_KEY"] = ""

class Resposta(BaseModel):
    resposta: str

    def __repr__(self):
        return f"Resposta({self.resposta})"

agente_rag = Agent(
    model="gpt-4o-mini",
    result_type=Resposta,
    system_prompt=""
)

contexto = input("Descreva aqui sobre o que deseja ensinar hoje:\n")

response = agente_rag.run_sync(
    user_prompt=f"Com base nisso:\n{contexto}\n\nGere 5 perguntas de múltipla escolha. Aumente o nível de dificuldade a cada pergunta."
)

texto_perguntas = response.data.resposta.strip()
print("\n📘 Perguntas geradas:\n", texto_perguntas)

form_metadata = {
    "info": {
        "title": f"Formulário: {contexto}",
        "documentTitle": f"Aula - {contexto}"
    }
}
form = service.forms().create(body=form_metadata).execute()
form_id = form["formId"]
print("\n✅ Formulário criado:", form["responderUri"])

blocos = re.split(r'\n(?=\d+\.\s)', texto_perguntas)
requests = []

for bloco in blocos:
    bloco = bloco.strip()
    if not bloco:
        continue

    linhas = bloco.split("\n")
    titulo = linhas[0].strip().replace("\n", " ")

    opcoes = []
    for linha in linhas[1:]:
        linha = linha.strip().replace("\n", " ")
        match = re.match(r"[a-dA-D]\)\s*(.+)", linha)
        valor = match.group(1).strip() if match else linha.strip()
        valor = valor.replace("\n", " ")
        opcoes.append({"value": valor})

    while len(opcoes) < 4:
        opcoes.append({"value": "Opção"})

    requests.append({
        "createItem": {
            "item": {
                "title": titulo,
                "questionItem": {
                    "question": {
                        "required": True,
                        "choiceQuestion": {
                            "type": "RADIO",
                            "options": opcoes[:4],
                            "shuffle": True
                        }
                    }
                }
            },
            "location": {
                "index": 0
            }
        }
    })

# === ENVIA AS QUESTÕES PARA O FORMULÁRIO ===
service.forms().batchUpdate(formId=form_id, body={"requests": requests}).execute()
print("\n✅ Perguntas adicionadas com sucesso!")
print("🔗 Link do formulário:", form["responderUri"])