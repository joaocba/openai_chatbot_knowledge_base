"""
The given code is a Python Flask web application that uses a chatbot to respond to user inputs.

The code imports Flask, request, render_template, and logging modules. Then it defines a root route (/) to render an HTML template, and a second route (/get) to receive user inputs via HTTP GET method.

The code then initializes a Flask app and defines the app routes. When the user sends a message to the chatbot by accessing the /get route, the code appends the message to a conversation array, generates a response using a chatbot_qa function, appends the response to the conversation array, and returns the response to the user.


This code imports the logging, Flask, request, and render_template modules. It also imports two functions "generate_datablocks_embeddings" and "chatbot_qa" from the "engine" module.

This code prepares data blocks embeddings and creates a Flask application. Two route decorators are defined in this code "index" and "get_bot_response".

The "index" function renders the "index.html" template when the route of the application is accessed. The "get_bot_response" function receives the input from the user, generates a response using the "chatbot_qa" function, and returns it to the user. The "chatbot_qa" function returns a response to the user based on the input.

The "if name == 'main':" statement is used to run the Flask application. When this code is executed, it starts a web server and the Flask application is served on it.





This code sets up a simple web interface for a chatbot using Flask.

In the first few lines, modules including logging, Flask and the engine module are imported. The engine module contains the functions generate_datablocks_embeddings and chatbot_qa which are used to generate embeddings for the content and to generate responses to user inputs respectively.

After importing the necessary modules, the logging level is set to debug using basicConfig() method. The generate_datablocks_embeddings() function is called to create embeddings for the content and a log message using the logging.info() method is printed indicating that the data has been prepared.

The Flask() method is called to create an instance of the Flask application and saved to the variable app. The @app.route decorator is then used to create two routes:

/ - bound to the index() function, which returns an HTML template with the chatbot interface.
/get- bound to the get_bot_response() function, which retrieves the user's input using the request.args.get() method, generates a chatbot response using the chatbot_qa method and returns the response to the user.
Lastly, the script specifies that the app should run if the name of our script is __main__, using the app.run() method. This allows us to execute the script as a standalone program. Overall, this code provides the foundation to create a simple chatbot interface using Flask and a chatbot engine with the functions generate_datablocks_embeddings and chatbot_qa.
"""


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
