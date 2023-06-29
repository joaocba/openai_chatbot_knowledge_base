"""
This code defines functions and implements an app that acts as a chatbot. The chatbot uses GPT-3 from OpenAI to generate responses to user inputs. The code starts by importing necessary libraries and setting up some variables. It then defines a main function that generates embeddings and runs a while loop to take user inputs, generate responses and append them to a list of messages exchanged between the user and the chatbot. The chatbot selects the appropriate messages to send to GPT-3, given a token limit of 3500, including system prompts and knowledge from previously answered questions. The chatbot then sends the messages to GPT-3 and returns the response. The code also includes functions to get and generate embeddings for the DATABLOCKS included in the chatbot's knowledge base.




The code starts by importing the necessary libraries and setting up some variables. Then, the code defines the following functions:

main(): runs the entire app by generating embeddings and running a while loop to take user inputs, generate responses, and append them to a list of messages exchanged between the user and the chatbot.

chatbot_qa(): the function runs the while loop that listens to user inputs and responds accordingly. It selects the appropriate messages to send to GPT-3, given a token limit of 3500, including system prompts and knowledge from previously answered questions. It applies the system initial prompt plus data from DATABLOCKS. It truncates the system message to reach an optimal 2500 tokens. It also looped in reverse over all messages (except the last one which is already added) and adds as many as possible without exceeding the token limit of 3500.

get_embedding(): accepts a string of text and returns a list of embeddings for that text.

generate_datablock_embeddings(): creates embeddings for the DATABLOCKS included in the chatbot's knowledge base. The function loops over all DATABLOCKS and verifies if the embedding for the datablock file already exists. If the embedding exists, the function will load the embedding from the file. If the embedding does not exist, the function will create the embedding and save it to a file.

All of the above functions work together to operate the chatbot using GPT-3 from OpenAI to generate respones for user input text.






This is a Python code consisting of a chatbot using the OpenAI API to provide an intelligent response system to user queries. The chatbot_qa function is the main function that receives user input as question, compares it with the available knowledge using the cosine similarity technique and generates text prompts/messages.

After generating questions' embeddings, the function compares them with the embeddings of available knowledge. Once the highest score is found, the corresponding datablock is selected. The function then retrieves information from this datablock and combines it with previous datablocks to provide additional context. It then applies a prompt of the system to this knowledge to teach the system its role.

The function aggregates messages from the user and the system and applies several logic checks on these messages, including token count and message length, before sending them to the GPT model via the API. It gets a response from the model and prints the response.

The code also includes two helper functions - get_embedding(text) and generate_datablocks_embeddings() to create embeddings for input texts and data blocks respectively. The latter function uses data from local text files of questions and answers to generate embeddings for local knowledge.

The functions exhibit the use of NLP techniques such as cosine similarity and text embeddings to provide a sophisticated chatbot capable of providing a reasonable response system.






The code contains three functions:

chatbot_qa(user_input_message): This is the main function that receives a message from the user and returns a response from the chatbot.

The function first creates an embedding for the user's input (question) using the get_embedding() helper function. It then compares this input embedding with the embeddings of available knowledge using cosine similarity and selects the datablock with the highest score.

The function then retrieves information from this datablock and combines it with previous datablocks to provide additional context. It then applies a prompt of the system (including the selected knowledge) to teach the system its role.

The function aggregates messages from the user and the system and applies several logic checks on these messages, including token count and message length, before sending them to the GPT model via the OpenAI API. It gets a response from the model and returns the response.

get_embedding(text: str) -> list[float]: This is a helper function that takes a string as input and returns its embedding using the OpenAI API.

generate_datablocks_embeddings() -> None: This is another helper function that generates embeddings for local knowledge (stored in text files) and saves them as JSON files.
"""

# Importar librarias necessárias
import glob
import json
import logging
import math
import os
import pathlib
import sys
import openai
from dotenv import load_dotenv
from openai.embeddings_utils import cosine_similarity
from _config import SYSTEM_PROMPT, count_tokens_from_messages

load_dotenv()

# Definir a chave API da OpenAI
openai.api_key = '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'

# Definir dicionários
DATABLOCKS: dict[str, str] = {}
EMBEDDINGS: dict[str, list[float]] = {}
MESSAGES: list[dict[str, str]] = []
DATABLOCKS_MATCH: list[str] = []

# Numero maximo de blocos de dados aceites para somar conhecido
MAX_DATABLOCKS = 5


# Função de pergunta e resposta chatbot
def chatbot_qa(user_input_message) -> None:
    while True:
        logging.info("Para sair pressione Enter novamente ou escreva 'Q'")
        question = user_input_message.strip()
        if not question or question == "Q":
            logging.info("A fechar, obrigado!")
            sys.exit(0)

        # Criar embedding para a questao do utilizador
        question_embeddings = get_embedding(question)
        embeddings_scores: dict[str, float] = {}
        logging.debug("A comparar embedding da pergunta com o conhecimento..")

        # Comparar embeddings da pergunta do utilizador com o conhecimento disponivel,
        # uso da técnica de simililariedade de cosseno
        for datablock, datablock_embeddings in EMBEDDINGS.items():
            embeddings_scores[datablock] = cosine_similarity(
                question_embeddings, datablock_embeddings
            )

        embeddings_scores_sorted = sorted(
            list(embeddings_scores.items()), key=lambda score: score[1], reverse=True
        )

        match_datablock = embeddings_scores_sorted[0][0]
        DATABLOCKS_MATCH.append(match_datablock)

        logging.debug("Embeddings scores: %s", embeddings_scores_sorted)
        logging.debug("A procurar resposta em: %s", match_datablock)

        # Usar o datablock com maior score de similariedade
        knowledge_count = 1
        knowledge = DATABLOCKS[match_datablock]

        # Adicionar o datablock anterior para o modelo entender o seguimento da conversa
        last_match_datablock = DATABLOCKS_MATCH[-1]
        if last_match_datablock != match_datablock:
            knowledge += f"\n{last_match_datablock}"

        # Adicionar todos os scores dos DATABLOCKS com valor
        # acima de 0.85 ao MAX_DATABLOCKS para somar conhecimento
        for datablock, score in embeddings_scores_sorted[1:]:
            if score >= 0.80:
                knowledge += f"\n{DATABLOCKS[datablock]}"
                knowledge_count += 1
            if knowledge_count > MAX_DATABLOCKS:
                break

        logging.debug("Conhecimento preparado de %s DATABLOCKS", knowledge_count)

        # Agregar a pergunta ao prompt do modelo GPT
        MESSAGES.append({"role": "user", "content": question})

        # Aplicar o prompt de sistema para o modelo GPT entender o seu papel
        system_prompt = f"{SYSTEM_PROMPT}{knowledge}"
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]

        # Separar a prompt de sistema para obter um numero maximo de tokens (ideal < 2500)
        knowledge_tokens_count = count_tokens_from_messages(messages)
        while True:
            if knowledge_tokens_count <= 2500:
                break
            knowledge_words = messages[0]["content"].split(" ")
            knowledge_words_count = len(knowledge_words)
            messages[0]["content"] = " ".join(
                knowledge_words[
                    : math.floor(
                        (knowledge_words_count * 2500) / knowledge_tokens_count
                    )
                ]
            )
            knowledge_tokens_count = count_tokens_from_messages(messages)

        # Agregar a mensagem de utilizador
        messages.append(MESSAGES[-1])

        # Correr de forma reversa todas as mensagens (excepto a ultima já adiciona)
        # e adicionar quantas possiveis sem quebrar o limite de 3500 tokens permitido pela call da API
        for i, message in enumerate(MESSAGES[1:-10:-1]):
            if count_tokens_from_messages([*messages, message]) < 3500:
                messages.insert(len(messages) - (i + 1), message)

        logging.debug("Mensagens a ser enviadas à OpenAI: %s", messages)
        logging.debug("Total de mensagens: %s", len(messages))

        # Obter resposta do modelo da OpenAI
        answer = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,
        )
        response_text = answer["choices"][0]["message"]["content"]
        MESSAGES.append({"role": "assistant", "content": response_text})

        # Print da resposta
        print(f"> {response_text}")
        return response_text


# Função para obter embeddings com o modelo da OpenAI
def get_embedding(text: str) -> list[float]:
    result = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return result["data"][0]["embedding"]


# Função para gerar embeddings de dados locais
def generate_datablocks_embeddings() -> None:
    logging.info("A verificar DATABLOCKS...")
    dirname = os.path.dirname(__file__)

    # Definir directorio e tipo de ficheiro
    DATABLOCKS = glob.glob(os.path.join(dirname, "data", "*.txt"))

    for datablock in DATABLOCKS:
        datablock_name = pathlib.Path(datablock).stem

        # Verificar se já existem embeddings criados
        if os.path.isfile(f"embeddings/{datablock_name}.json"):
            logging.info(
                "Datablock '%s' embeddings já foram criados. A saltar..", datablock_name
            )

            with open(datablock, encoding="UTF-8") as f:
                DATABLOCKS[datablock_name] = f.read()

            with open(f"embeddings/{datablock_name}.json", encoding="UTF-8") as f:
                EMBEDDINGS[datablock_name] = json.loads(f.read())

            continue

        logging.info(
            "Datablock '%s' embeddings não foram ainda criados. A criar..", datablock_name
        )

        # Criar embeddings para o conteudo
        with open(datablock, encoding="UTF-8") as f:
            datablock_content = f.read()
            DATABLOCKS[datablock_name] = datablock_content
            datablock_embeddings = get_embedding(datablock_content)
            EMBEDDINGS[datablock_name] = datablock_embeddings

        with open(f"embeddings/{datablock_name}.json", "w", encoding="UTF-8") as f:
            f.write(json.dumps(datablock_embeddings))

        logging.info("  - Feito")


