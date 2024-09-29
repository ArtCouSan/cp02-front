import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO

# Carregar o modelo
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Carregar os dados do DataFrame
df = pd.read_csv('./deploy-streamlit/Xtest.csv')

# Certifique-se de que o DataFrame possui uma coluna chamada 'image_url' com os links das imagens
# df['image_url'] = 'http://link_to_image.jpg'  # Caso não tenha, adicione a coluna manualmente

# Título da aplicação
st.title("Movie Recommendation System")

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

        cols = st.columns(5)  # Cria 5 colunas para os filmes recomendados

        for idx, (_, row) in enumerate(recommended_movies.iterrows()):
            with cols[idx % 5]:  # Distribui os filmes nas colunas
                # Obter a imagem do filme usando o link contido na coluna 'image_url'
                try:
                    response = requests.get(row['image_url'])
                    img = Image.open(BytesIO(response.content))
                except:
                    img = Image.open('default_image.jpg')  # Imagem padrão caso falhe

                st.image(img, use_column_width=True)
                st.markdown(f"**{row['title_en']}**")
                st.caption(row['synopsis'])
    else:
        st.warning("There are no movies in the same cluster.")
