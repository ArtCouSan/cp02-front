import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO

# Configurações de estilo
st.set_page_config(layout="wide")  # Layout de página mais largo

# Incluir fontes personalizadas no cabeçalho
st.markdown(
    """
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
    </head>
    """, 
    unsafe_allow_html=True
)

# Aplicar CSS personalizado para tema Netflix
st.markdown(
    """
    <style>
    /* Background geral */
    .block-container {
        padding: 1rem 1rem 10rem;
    }
    .stApp {
        background-color: #141414;  /* Fundo preto */
    }
    /* Estilo do título */
    .css-10trblm {
        color: #e50914;  /* Vermelho Netflix */
        font-family: 'Bebas Neue', sans-serif; /* Fonte Bebas Neue */
        font-size: 3em;
        text-align: center;
        font-weight: bold;
    }
    /* Subtítulo */
    .css-16huue1 {
        color: #ffffff;
        font-family: 'Roboto', sans-serif; /* Fonte Roboto */
        text-align: center;
        font-size: 1.2em;
    }
    /* Caixa de entrada de texto */
    .css-1msw4hw {
        background-color: #333;
        color: white;
        border-color: #e50914;  /* Borda vermelha */
    }
    /* Botão */
    .css-1cpxqw2 {
        background-color: #e50914;  /* Fundo vermelho */
        color: white;
        font-family: 'Roboto', sans-serif; /* Fonte Roboto */
        font-size: 1em;
        font-weight: bold;
        border: none;
    }
    .css-1cpxqw2:hover {
        background-color: #b20710;  /* Vermelho mais escuro ao passar o mouse */
    }
    /* Estilo das imagens */
    .stImage {
        border: 2px solid #e50914;  /* Borda vermelha */
        border-radius: 5px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.5);  /* Sombra */
    }
    /* Texto centralizado */
    .movie-title {
        color: #e50914;
        font-family: 'Bebas Neue', sans-serif; /* Fonte Bebas Neue */
        font-size: 1.5em;
        text-align: center;
        font-weight: bold;
    }
    /* Legenda */
    .movie-caption {
        color: #ffffff;
        font-family: 'Roboto', sans-serif; /* Fonte Roboto */
        text-align: center;
        font-size: 0.9em;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Carregar o modelo
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Carregar os dados do DataFrame
df = pd.read_csv('../deploy-streamlit/Xtest.csv', delimiter=';', encoding='utf-8')

# Título da aplicação
st.title("IA Flix")

# Subtítulo
st.markdown("<div class='css-16huue1'>Discover movies that match your interests!</div>", unsafe_allow_html=True)

# Entrada do usuário para a sinopse
new_synopsis = st.text_area("Enter the new movie synopsis:", "american car designer carroll shelby driver")

if st.button("Recommend Movies"):
    # Codificar a nova sinopse
    new_embedding = model.encode(new_synopsis)

    # Codificar sinopses existentes
    df['embeddings'] = df['synopsis'].apply(lambda x: model.encode(x))
    # Converter embeddings existentes em uma matriz NumPy
    Xtest = np.vstack(df['embeddings'].values)  # Certifique-se de que df['embeddings'] seja uma lista de arrays

    # Calcular similaridade coseno
    similarities = cosine_similarity([new_embedding], Xtest)[0]  # Retorna um array de similaridades

    # Obter o índice do filme mais semelhante
    most_similar_index = similarities.argmax()  # Índice do valor mais alto de similaridade

    # Obter o cluster do filme mais semelhante
    target_cluster = df.iloc[most_similar_index]['cluster']

    # Filtrar filmes que pertencem ao mesmo cluster
    cluster_movies = df[df['cluster'] == target_cluster]

    if not cluster_movies.empty:  # Verificar se há filmes no cluster
        # Calcular similaridade coseno apenas para filmes no mesmo cluster
        cluster_embeddings = np.vstack(cluster_movies['embeddings'].values)
        cluster_similarities = cosine_similarity([new_embedding], cluster_embeddings)[0]

        # Obter os índices dos 5 filmes mais semelhantes no cluster
        top_indices_in_cluster = cluster_similarities.argsort()[-5:][::-1]  # Índices dos 5 valores mais altos, em ordem decrescente

        # Obter os filmes recomendados
        recommended_movies = cluster_movies.iloc[top_indices_in_cluster]

        # Exibir resultados em um layout de grid
        st.subheader("Recommended Movies")

        # Dividir a tela em colunas para um layout em grid
        cols = st.columns(5)  # Cria 5 colunas para os filmes recomendados

        for idx, (_, row) in enumerate(recommended_movies.iterrows()):
            with cols[idx % 5]:  # Distribui os filmes nas colunas
                try:
                    # Carregar a imagem do filme usando o link contido na coluna 'imgs'
                    response = requests.get(row['imgs'])
                    img = Image.open(BytesIO(response.content))
                except Exception as e:
                    st.error(f"Erro ao carregar a imagem: {e}")
                    img = Image.open('default_image.jpg')  # Imagem padrão caso falhe

                # Exibir a imagem com estilo personalizado
                st.image(img, use_column_width=True, caption="", output_format="auto", clamp=False)
                st.markdown(f"<div class='movie-title'>{row['title_en']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='movie-caption'>{row['synopsis'][:100]}...</div>", unsafe_allow_html=True)

    else:
        st.warning("There are no movies in the same cluster.")
