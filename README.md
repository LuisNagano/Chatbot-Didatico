# Chatbot Usando Python De Forma Didática

## Índice

1. [Visão Geral do Projeto](#1-visão-geral-do-projeto)
2. [Funcionalidades](#2-funcionalidades)
3. [Tecnologias Utilizadas](#3-tecnologias-utilizadas)
4. [Estrutura do Projeto](#4-estrutura-do-projeto)
5. [Explicação Detalhada](#5-explicação-detalhada)
    - [1. Importação de Bibliotecas](#51-importação-de-bibliotecas)
    - [2. Tokenização de Texto](#52-tokenização-de-texto)
    - [3. Definição de Intenções](#53-definição-de-intenções)
    - [4. Transformação das Frases de Entrada](#54-transformação-das-frases-de-entrada)
    - [5. Treinamento do Classificador de Regressão Logística](#55-treinamento-do-classificador-de-regressão-logística)
    - [6. Função de Resposta do Chatbot](#56-função-de-resposta-do-chatbot)
    - [7. Teste da Função de Resposta](#57-teste-da-função-de-resposta)
    - [8. Loop de Interação com o Usuário](#58-loop-de-interação-com-o-usuário)
    - [9. Integração com Flask](#59-integração-com-flask)
    - [10. Execução do App Flask](#510-execução-do-app-flask)
6. [Contribuições](#6-contribuições)
7. [Licença](#7-licença)

---

## 1. Visão Geral do Projeto

Este projeto demonstra a criação de um chatbot inteligente usando Python. O chatbot é capaz de entender entradas dos usuários e responder de maneira apropriada, aproveitando técnicas de NLP e ML. É uma implementação robusta e completa, desde o pré-processamento de dados até a implantação do chatbot usando Flask.

## 2. Funcionalidades

- **Processamento de Linguagem Natural**: Tokenização e processamento de texto eficiente usando `nltk`.
- **Aprendizado de Máquina**: Implementação de Regressão Logística para classificação de intenções.
- **Interface Interativa**: Loop contínuo de interação com o usuário e interface web baseada em Flask.
- **Intenções Personalizáveis**: Fácil definição e extensão de intenções para o chatbot.

## 3. Tecnologias Utilizadas

- **Python**: A linguagem principal para a implementação.
- **nltk**: Para processamento e tokenização de texto.
- **scikit-learn**: Para extração de características (`TfidfVectorizer`) e aprendizado de máquina (`LogisticRegression`).
- **Flask**: Para implantação do chatbot como uma aplicação web.

## 4. Estrutura do Projeto

```plaintext
Chatbot-Didatico/
├── app_flask.py
├── templates/
│   └── index.html
└── README.md
```

- **app_flask.py**: O arquivo principal da aplicação Flask.
- **templates/index.html**: Template HTML para a interface web.
- **README.md**: Arquivo README do projeto.

## 5. Explicação Detalhada

### 5.1. Importação de Bibliotecas

```python
import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
```

### 5.2. Tokenização de Texto

```python
nltk.download("punkt")
```

### 5.3. Definição de Intenções

As intenções são definidas como uma lista de dicionários, cada um representando uma intenção específica com padrões e respostas associadas.

```python
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    ...
]
```

### 5.4. Transformação das Frases de Entrada

Usando `TfidfVectorizer` para converter frases de entrada em vetores numéricos.

```python
vectorizer = TfidfVectorizer()
```

### 5.5. Treinamento do Classificador de Regressão Logística

Treinando o classificador com vetores TF-IDF.

```python
classifier = LogisticRegression(random_state=0, max_iter=10000)
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)
x = vectorizer.fit_transform(patterns)
y = tags
classifier.fit(x, y)
```

### 5.6. Função de Resposta do Chatbot

Definindo uma função para gerar respostas com base na entrada do usuário.

```python
def chatbot_response(text):
    input_text = vectorizer.transform([text])
    tag = classifier.predict(input_text)[0]
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent['responses'])
```

### 5.7. Teste da Função de Resposta

```python
response = chatbot_response("What are the helps you provide?")
print(response)
```

### 5.8. Loop de Interação com o Usuário

```python
while True:
    query = input("User-> ")
    response = chatbot_response(query)
    print("Chatbot-> {}".format(response))
```

### 5.9. Integração com Flask

Criando uma interface web usando Flask.

```python
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        response = chatbot_response(user_input)
        return render_template("index.html", user_input=user_input, response=response)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
```

### 5.10. Execução do App Flask

Siga estes passos para executar a aplicação Flask:

1. Abra o prompt de comando ou Anaconda.
2. Navegue até o diretório do projeto.
3. Execute o app Flask:
    ```bash
    python app_flask.py
    ```
4. Abra o URL fornecido no seu navegador.

## 6. Contribuições

Contribuições são bem-vindas! Por favor, faça um fork deste repositório e envie um pull request para quaisquer melhorias, correções de bugs ou novas funcionalidades.

## 7. Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
