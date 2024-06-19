import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from google.cloud import storage
from dotenv import load_dotenv

# Memuat variabel lingkungan dari file .env
load_dotenv()

# Menggunakan variabel lingkungan
model_path = os.environ.get("MODEL_PATH")
csv_path = os.environ.get("CSV_PATH")
bucket_name = os.environ.get("BUCKET_NAME")
google_application_credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

# Set the environment variable for Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_application_credentials

# Verifikasi versi TensorFlow dan Keras
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

app = FastAPI()

# Function to download the model from GCS
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

# Mengunduh model dari GCS
try:
    local_model_path = "model.h5"
    download_blob(bucket_name, model_path, local_model_path)
    model = load_model(local_model_path)
except Exception as e:
    import logging
    logging.error(f"Error loading model: {e}")
    raise

# Load the CSV file
file_path = csv_path
data = pd.read_csv(file_path)

# Data preprocessing
data['product_name'] = data['product_name'].apply(lambda x: x.replace(u'\xa0', u' '))
data['product_name'] = data['product_name'].apply(lambda x: x.replace('\u200a', ' '))
data['product_name'] = data['product_name'].str.lower()
corpus = data['product_name']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['product_name'])
total_words = len(tokenizer.word_index) + 1

# Function n-gram
def n_gram_seqs(corpus, tokenizer):
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequences = token_list[:i+1]
            input_sequences.append(n_gram_sequences)
    return input_sequences

input_sequences = n_gram_seqs(corpus, tokenizer)
max_sequence_len = max([len(x) for x in input_sequences])

# Function padded
def pad_seqs(input_sequences, maxlen):
    padded_sequences = pad_sequences(input_sequences, maxlen=maxlen)
    return padded_sequences

input_sequences = pad_seqs(input_sequences, max_sequence_len)

# Embedding input
def get_embeddings(model, tokenizer, texts, max_sequence_len):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len-1)
    embeddings = model.predict(padded_sequences)
    return embeddings

product_embeddings = get_embeddings(model, tokenizer, data['product_name'], max_sequence_len)

def recommend_products(user_input, model, tokenizer, product_embeddings, data, max_sequence_len, page=1, per_page=50):
    input_seq = tokenizer.texts_to_sequences([user_input])
    input_padded = pad_sequences(input_seq, maxlen=max_sequence_len-1)
    input_embedding = model.predict(input_padded)

    similarities = cosine_similarity(input_embedding, product_embeddings)
    similar_indices = similarities.argsort()[0][::-1]

    recommendations = data.iloc[similar_indices][['product_name', 'rent_price', 'url_photo', 'product_id', 'rating']]
    
    start = (page - 1) * per_page
    end = start + per_page

    total_recommendations = len(recommendations)
    paginated_recommendations = recommendations.iloc[start:end]

    return paginated_recommendations, total_recommendations

@app.get("/search")
async def recommend(query: str = Query(..., description="The search query for recommendations"), 
                    page: int = 1, 
                    per_page: int = 30):
    if not query:
        raise HTTPException(status_code=400, detail="Keyword is required")

    recommendations, total_recommendations = recommend_products(
        query, model, tokenizer, product_embeddings, data, max_sequence_len, page, per_page
    )
    recommendations_dict = recommendations.to_dict(orient='records')

    # Tambahkan 'f' di belakang nilai rating
    for recommendation in recommendations_dict:
        if 'rating' in recommendation:
            recommendation['rating'] = f"{recommendation['rating']}f"
        if 'rent_price' in recommendation:
            recommendation['rent_price'] = round(recommendation['rent_price'])

    return JSONResponse(content={
        'page': page,
        'per_page': per_page,
        'total': total_recommendations,
        'products': recommendations_dict
    })

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 8080))
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        import logging
        logging.error(f"Error occurred: {e}")
        raise
