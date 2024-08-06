# Chatbot usando python de forma didática

---

Este projeto demonstra a implementação completa de um chatbot usando Python, incorporando técnicas de Processamento de Linguagem Natural (NLP) e Aprendizado de Máquina (ML). O chatbot é projetado para interagir com os usuários, entender suas consultas e fornecer respostas apropriadas. Este README irá guiá-lo através do projeto, sua estrutura, como configurá-lo e executá-lo.

---

## Índice

1. [Visão Geral do Projeto](#1-visão-geral-do-projeto)
2. [Funcionalidades](#2-funcionalidades)
3. [Tecnologias Utilizadas](#3-tecnologias-utilizadas)
4. [Configuração e Instalação](#4-configuração-e-instalação)
5. [Estrutura do Projeto](#5-estrutura-do-projeto)
6. [Explicação Detalhada](#6-explicação-detalhada)
    - [1. Importação de Bibliotecas](#61-importação-de-bibliotecas)
    - [2. Tokenização de Texto](#62-tokenização-de-texto)
    - [3. Definição de Intenções](#63-definição-de-intenções)
    - [4. Transformação das Frases de Entrada](#64-transformação-das-frases-de-entrada)
    - [5. Treinamento do Classificador de Regressão Logística](#65-treinamento-do-classificador-de-regressão-logística)
    - [6. Função de Resposta do Chatbot](#66-função-de-resposta-do-chatbot)
    - [7. Teste da Função de Resposta](#67-teste-da-função-de-resposta)
    - [8. Loop de Interação com o Usuário](#68-loop-de-interação-com-o-usuário)
    - [9. Integração com Flask](#69-integração-com-flask)
    - [10. Execução do App Flask](#610-execução-do-app-flask)
7. [Contribuições](#7-contribuições)
8. [Licença](#8-licença)

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

## 4. Configuração e Instalação

### Pré-requisitos

Certifique-se de ter o Python instalado no seu sistema. Você pode baixá-lo em [python.org](https://www.python.org/).

### Passos de Instalação

1. **Clone o Repositório**:
    ```bash
    git clone https://
    cd End-to-End-Chatbot
    ```

2. **Crie um Ambiente Virtual**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows use `venv\Scripts\activate`
    ```

3. **Instale as Bibliotecas Necessárias**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Execute o App Flask**:
    ```bash
    python app_flask.py
    ```

5. **Acesse o Chatbot**:
    Abra seu navegador e navegue até `http://127.0.0.1:5001/`.

## 5. Estrutura do Projeto

```plaintext
End-to-End-Chatbot/
├── app_flask.py
├── intents.json
├── requirements.txt
├── templates/
│   └── index.html
└── README.md
```

- **app_flask.py**: O arquivo principal da aplicação Flask.
- **intents.json**: Arquivo JSON contendo as intenções predefinidas.
- **requirements.txt**: Lista de dependências do projeto.
- **templates/index.html**: Template HTML para a interface web.
- **README.md**: Arquivo README do projeto.

## 6. Explicação Detalhada

### 6.1. Importação de Bibliotecas

```python
import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
```

### 6.2. Tokenização de Texto

```python
nltk.download("punkt")
```

### 6.3. Definição de Intenções

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

### 6.4. Transformação das Frases de Entrada

Usando `TfidfVectorizer` para converter frases de entrada em vetores numéricos.

```python
vectorizer = TfidfVectorizer()
```

### 6.5. Treinamento do Classificador de Regressão Logística

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

### 6.6. Função de Resposta do Chatbot

Definindo uma função para gerar respostas com base na entrada do usuário.

```python
def chatbot_response(text):
    input_text = vectorizer.transform([text])
    tag = classifier.predict(input_text)[0]
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent['responses'])
```

### 6.7. Teste da Função de Resposta

```python
response = chatbot_response("What are the helps you provide?")
print(response)
```

### 6.8. Loop de Interação com o Usuário

```python
while True:
    query = input("User-> ")
    response = chatbot_response(query)
    print("Chatbot-> {}".format(response))
```

### 6.9. Integração com Flask

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

### 6.10. Execução do App Flask

Siga estes passos para executar a aplicação Flask:

1. Abra o prompt de comando ou Anaconda.
2. Navegue até o diretório do projeto.
3. Execute o app Flask:
    ```bash
    python app_flask.py
    ```
4. Abra o URL fornecido no seu navegador.

## 7. Contribuições

Contribuições são bem-vindas! Por favor, faça um fork deste repositório e envie um pull request para quaisquer melhorias, correções de bugs ou novas funcionalidades.

## 8. Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
