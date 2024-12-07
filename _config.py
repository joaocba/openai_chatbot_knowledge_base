import tiktoken

# Definir o papel do sistema
SYSTEM_PROMPT = (
    "You are a chat bot who is supposed to help users with "
    "companies details and reviews related questions. You should "
    "politely decline answering any question not related to companies "
    "details and reviews. You should always answer in "
    "the same language as the question is. Your only source of knowledge is following text: "
)

# Função para calcular a contagem de tokens por mensagem
def count_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # Caso não utilize o modelo GPT 3.5 Turbo retorna erro
    if model != "gpt-3.5-turbo":
        raise NotImplementedError(
            f"count_tokens_from_messages() não está implementado no modelo {model}."
            " Veja https://github.com/openai/openai-python/blob/main/chatml.md para "
            "mais informações de como as mensagens são convertidas em tokens."
        )

    num_tokens = 0
    for message in messages:
        # Cada mensagem contem o seguinte corpus: <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                # Subtrair o papel do sistema da contagem
                num_tokens -= 1
    # Incrementar o papel do assistente <im_start>assistant
    num_tokens += 2
    return num_tokens
