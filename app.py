import logging
from flask import Flask, request, render_template
from engine import generate_datablocks_embeddings, chatbot_qa

# Definir nivel de logger
logging.basicConfig(level=logging.DEBUG)

# Criar embeddings para o conteudo
generate_datablocks_embeddings()
logging.info("Dados preparados. Pronto!")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    # Obter o input to utilizador
    user_input = request.args.get("msg")

    if user_input:
        # Gerar resposta ao input do utilzador
        response = chatbot_qa(user_input)
    else:
        response = "Peço desculpa, não entendi."

    return response

if __name__ == "__main__":
    app.run()
