"""
REKOMENDASI BERDASARKAN DATASET LOKAL - MODEL BARU
"""
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import ast
import requests
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import time

# ============================
# Konfigurasi Halaman
# ============================
st.set_page_config(page_title="Anime Recommender - New Model", layout="centered")

# ============================
# Load Dataset dan Model Baru
# ============================
@st.cache_resource
def load_new_resources():
    try:
        # Load model dan komponen baru
        recommendation_model = load_model("anime_recommendation_model.h5")
        
        with open("model_components.pkl", "rb") as f:
            components = pickle.load(f)
            scaler = components['scaler']
            tfidf_vectorizer = components['tfidf_vectorizer']
            svd_model = components['svd_model']
            label_encoder = components['label_encoder']
            feature_columns = components['feature_columns']
            similarity_matrix = components['similarity_matrix']
        
        # Load dataset yang sudah diproses
        processed_data = pd.read_csv("processed_anime_data.csv")
        
        # Load dataset asli untuk informasi tambahan
        original_data = pd.read_csv("anime_full_dataset_cleaned_synopsis.csv")
        
        return (recommendation_model, scaler, tfidf_vectorizer, svd_model, 
                label_encoder, feature_columns, similarity_matrix, 
                processed_data, original_data)
                
    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan: {e}")
        st.info("Pastikan file model sudah ada: anime_recommendation_model.h5, model_components.pkl, processed_anime_data.csv")
        return None

# Load resources
resources = load_new_resources()
if resources is None:
    st.stop()

(model, scaler, tfidf_vectorizer, svd_model, label_encoder, 
 feature_columns, similarity_matrix, processed_data, original_data) = resources

# ============================
# Kelas Sistem Rekomendasi Baru
# ============================
class NewAnimeRecommendationSystem:
    def __init__(self, processed_data, original_data, similarity_matrix, model, 
                 scaler, tfidf_vectorizer, svd_model, label_encoder, feature_columns):
        self.processed_data = processed_data
        self.original_data = original_data
        self.similarity_matrix = similarity_matrix
        self.model = model
        self.scaler = scaler
        self.tfidf_vectorizer = tfidf_vectorizer
        self.svd_model = svd_model
        self.label_encoder = label_encoder
        self.feature_columns = feature_columns

    def safe_literal_eval(self, x):
        """Safely evaluate string representations of lists"""
        try:
            return ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else []
        except:
            return []

    def get_recommendations_by_mal_id(self, mal_id, num_recommendations=10, similarity_threshold=0.3):
        """Get anime recommendations based on MAL ID"""
        try:
            # Find anime in processed data
            anime_row = self.processed_data[self.processed_data['mal_id'] == mal_id]
            if anime_row.empty:
                return {"error": "Anime dengan MAL ID tersebut tidak ditemukan dalam dataset"}
            
            idx = anime_row.index[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.similarity_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Filter by similarity threshold and get recommendations
            recommendations = []
            for i, score in sim_scores[1:]:  # Skip the anime itself
                if score < similarity_threshold:
                    continue
                    
                if len(recommendations) >= num_recommendations:
                    break
                
                # Get anime info from both datasets
                rec_anime = self.processed_data.iloc[i]
                original_anime = self.original_data[self.original_data['mal_id'] == rec_anime['mal_id']]
                
                if not original_anime.empty:
                    original_anime = original_anime.iloc[0]
                    
                    # Process genres
                    genres = self.safe_literal_eval(original_anime.get('genres', '[]'))
                    genre_names = []
                    for g in genres:
                        if isinstance(g, dict):
                            genre_names.append(g.get('name', ''))
                        elif isinstance(g, str):
                            genre_names.append(g)
                    
                    # Process studios
                    studios = self.safe_literal_eval(original_anime.get('studios', '[]'))
                    studio_names = []
                    for s in studios:
                        if isinstance(s, dict):
                            studio_names.append(s.get('name', ''))
                        elif isinstance(s, str):
                            studio_names.append(s)
                    
                    # Process streaming platforms
                    streaming_platforms = self.safe_literal_eval(original_anime.get('streaming_platforms', '[]'))
                    streaming_links = []
                    for stream in streaming_platforms:
                        if isinstance(stream, dict):
                            name = stream.get('name', '')
                            url = stream.get('url', '')
                            if name:
                                streaming_links.append({"name": name, "url": url})
                    
                    anime_info = {
                        'title': str(original_anime.get('title_english', original_anime.get('title', 'Unknown'))),
                        'mal_id': int(rec_anime['mal_id']),
                        'score': float(original_anime.get('score', 0)),
                        'genres': genre_names,
                        'studios': studio_names,
                        'similarity_score': round(float(score) * 100, 2),
                        'synopsis': str(original_anime.get('synopsis', ''))[:400] + "...",
                        'image_url': str(original_anime.get('image_url', '')),
                        'trailer_url': str(original_anime.get('trailer_url', '')),
                        'rank': str(original_anime.get('rank', 'Tidak diketahui')),
                        'episodes': str(original_anime.get('episodes', 'Tidak diketahui')),
                        'streaming_links': streaming_links,
                        'type': str(original_anime.get('type', 'Unknown')),
                        'members': int(original_anime.get('members', 0)),
                        'favorites': int(original_anime.get('favorites', 0))
                    }
                    recommendations.append(anime_info)
            
            return {"recommendations": recommendations, "total_found": len(recommendations)}
            
        except Exception as e:
            return {"error": f"Terjadi kesalahan: {str(e)}"}

    def predict_anime_quality(self, mal_id):
        """Predict anime quality using the trained model"""
        try:
            anime_row = self.processed_data[self.processed_data['mal_id'] == mal_id]
            if anime_row.empty:
                return {"error": "Anime tidak ditemukan"}
            
            # Extract features for prediction
            anime_features = anime_row[self.feature_columns].fillna(0).values[0]
            
            # Scale features
            features_scaled = self.scaler.transform([anime_features])
            
            # Make prediction
            prediction = self.model.predict(features_scaled, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction))
            
            return {
                'predicted_category': self.label_encoder.classes_[predicted_class],
                'confidence': confidence,
                'all_probabilities': {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.label_encoder.classes_, prediction[0])
                }
            }
            
        except Exception as e:
            return {"error": f"Gagal memprediksi kualitas: {str(e)}"}

    def search_anime_by_title(self, title, limit=10):
        """Search anime by title in the dataset"""
        try:
            mask = self.original_data['title'].str.contains(title, case=False, na=False)
            if 'title_english' in self.original_data.columns:
                mask = mask | self.original_data['title_english'].str.contains(title, case=False, na=False)
            
            results = self.original_data[mask].head(limit)
            
            search_results = []
            for _, row in results.iterrows():
                search_results.append({
                    'mal_id': int(row['mal_id']),
                    'title': str(row.get('title_english', row.get('title', 'Unknown'))),
                    'score': float(row.get('score', 0)),
                    'type': str(row.get('type', 'Unknown'))
                })
            
            return search_results
            
        except Exception as e:
            return []

# Initialize new recommendation system
recommender = NewAnimeRecommendationSystem(
    processed_data, original_data, similarity_matrix, model, 
    scaler, tfidf_vectorizer, svd_model, label_encoder, feature_columns
)

# ============================
# Fungsi untuk mendapatkan info anime dari API
# ============================
def get_anime_info_api(mal_id):
    """Get anime info from Jikan API"""
    try:
        url = f"https://api.jikan.moe/v4/anime/{mal_id}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get("data", {})
            return {
                'title': data.get('title', 'Unknown'),
                'title_english': data.get('title_english', ''),
                'synopsis': data.get('synopsis', ''),
                'score': data.get('score', 0),
                'image_url': data.get('images', {}).get('jpg', {}).get('image_url', ''),
                'trailer_url': data.get('trailer', {}).get('url', ''),
                'genres': [g.get('name', '') for g in data.get('genres', [])],
                'studios': [s.get('name', '') for s in data.get('studios', [])]
            }
    except:
        pass
    return None

# ============================
# Streamlit UI
# ============================
st.title("🎌 Anime Recommender - Model Baru")
st.markdown(
    "Sistem rekomendasi anime menggunakan **Deep Learning Model** dengan fitur:\n"
    "- Content-based filtering dengan similarity matrix\n"
    "- Prediksi kualitas anime menggunakan neural network\n"
    "- Dataset lokal yang sudah diproses dengan TF-IDF dan SVD\n\n"
    "**Contoh MAL ID:** 5114 (Fullmetal Alchemist: Brotherhood), 9253 (Steins;Gate)"
)

# Tab untuk berbagai fitur
tab1, tab2, tab3 = st.tabs(["🔍 Search & Recommend", "🎯 Quality Prediction", "📊 Dataset Info"])

with tab1:
    st.header("Pencarian dan Rekomendasi Anime")
    
    # Search by title
    st.subheader("Cari Anime berdasarkan Judul")
    search_query = st.text_input("Masukkan judul anime", "")
    
    if search_query.strip():
        with st.spinner("Mencari anime..."):
            search_results = recommender.search_anime_by_title(search_query)
            
            if search_results:
                st.success(f"Ditemukan {len(search_results)} anime")
                for result in search_results:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{result['title']}** ({result['type']})")
                    with col2:
                        st.write(f"Score: {result['score']}")
                    with col3:
                        st.code(result['mal_id'])
            else:
                st.warning("Tidak ada anime yang ditemukan")
    
    st.divider()
    
    # Recommend by MAL ID
    st.subheader("Dapatkan Rekomendasi")
    mal_id_input = st.text_input("Masukkan MAL ID anime", "")
    
    col1, col2 = st.columns(2)
    with col1:
        num_recommendations = st.slider("Jumlah rekomendasi", 5, 20, 10)
    with col2:
        similarity_threshold = st.slider("Threshold kemiripan", 0.1, 0.9, 0.3)
    
    if mal_id_input.strip() and mal_id_input.isdigit():
        mal_id = int(mal_id_input)
        
        with st.spinner("Menganalisis dan mencari rekomendasi..."):
            # Get API info for reference anime
            api_info = get_anime_info_api(mal_id)
            
            if api_info:
                st.subheader(f"📌 Anime Referensi: {api_info['title']}")
                col1, col2 = st.columns([1, 3])
                with col1:
                    if api_info['image_url']:
                        st.image(api_info['image_url'], width=150)
                with col2:
                    st.write(f"**Score:** {api_info['score']}")
                    st.write(f"**Genres:** {', '.join(api_info['genres'])}")
                    st.write(f"**Studios:** {', '.join(api_info['studios'])}")
                
                if api_info['synopsis']:
                    with st.expander("Lihat Synopsis"):
                        st.write(api_info['synopsis'])
            
            # Get recommendations
            result = recommender.get_recommendations_by_mal_id(
                mal_id, num_recommendations, similarity_threshold
            )
            
            if "error" in result:
                st.error(result["error"])
            else:
                recommendations = result["recommendations"]
                st.success(f"✅ Ditemukan {result['total_found']} rekomendasi")
                
                if recommendations:
                    st.subheader("🎯 Rekomendasi Anime:")
                    
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"### {i}. {rec['title']}")
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if rec['image_url'] and rec['image_url'] != 'nan':
                                st.image(rec['image_url'], width=150)
                            else:
                                st.info("Gambar tidak tersedia")
                        
                        with col2:
                            st.markdown(f"⭐ **Score:** {rec['score']:.2f} &nbsp;&nbsp; 🔁 **Similarity:** {rec['similarity_score']}%")
                            st.markdown(f"**Type:** {rec['type']} &nbsp;&nbsp; **Episodes:** {rec['episodes']}")
                            st.markdown(f"**Rank:** {rec['rank']}")
                            st.markdown(f"**Studios:** {', '.join(rec['studios']) if rec['studios'] else 'Unknown'}")
                            st.markdown(f"**Genres:** {', '.join(rec['genres']) if rec['genres'] else 'Unknown'}")
                            st.markdown(f"**Members:** {rec['members']:,} &nbsp;&nbsp; **Favorites:** {rec['favorites']:,}")
                        
                        with st.expander("📖 Synopsis"):
                            st.write(rec['synopsis'])
                        
                        if rec['trailer_url'] and rec['trailer_url'] != 'nan':
                            st.video(rec['trailer_url'])
                        
                        if rec['streaming_links']:
                            st.markdown("**Streaming:**")
                            for stream in rec['streaming_links']:
                                if stream.get('url'):
                                    st.markdown(f"- [{stream['name']}]({stream['url']})")
                                else:
                                    st.markdown(f"- {stream['name']}")
                        
                        st.divider()
                else:
                    st.warning(f"Tidak ada rekomendasi yang ditemukan dengan similarity > {similarity_threshold}")

with tab2:
    st.header("Prediksi Kualitas Anime")
    st.write("Gunakan model deep learning untuk memprediksi kategori kualitas anime")
    
    predict_mal_id = st.text_input("MAL ID untuk prediksi kualitas", "")
    
    if predict_mal_id.strip() and predict_mal_id.isdigit():
        mal_id = int(predict_mal_id)
        
        with st.spinner("Memprediksi kualitas anime..."):
            prediction_result = recommender.predict_anime_quality(mal_id)
            
            if "error" in prediction_result:
                st.error(prediction_result["error"])
            else:
                st.success("Prediksi berhasil!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Kategori Prediksi", prediction_result['predicted_category'])
                with col2:
                    st.metric("Confidence", f"{prediction_result['confidence']:.2%}")
                
                st.subheader("Probabilitas untuk Setiap Kategori:")
                prob_data = prediction_result['all_probabilities']
                
                # Create a bar chart for probabilities
                prob_df = pd.DataFrame(list(prob_data.items()), columns=['Category', 'Probability'])
                prob_df = prob_df.sort_values('Probability', ascending=False)
                
                st.bar_chart(prob_df.set_index('Category'))
                
                # Show detailed probabilities
                for category, prob in prob_df.values:
                    st.write(f"**{category}:** {prob:.2%}")

with tab3:
    st.header("Informasi Dataset")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Anime dalam Dataset", len(processed_data))
        st.metric("Fitur yang Digunakan", len(feature_columns))
    with col2:
        st.metric("Kategori Kualitas", len(label_encoder.classes_))
        st.metric("Dimensi Similarity Matrix", f"{similarity_matrix.shape[0]} x {similarity_matrix.shape[1]}")
    
    st.subheader("Kategori Kualitas Anime:")
    for category in label_encoder.classes_:
        st.write(f"- {category}")
    
    st.subheader("Sample Data Processing:")
    if st.button("Tampilkan sample processed data"):
        st.dataframe(processed_data.head())

# ============================
# Footer
# ============================
st.markdown("---")
st.markdown(
    "**Anime Recommendation System** menggunakan Deep Learning dan Content-Based Filtering\n\n"
    "Model Features: TF-IDF + SVD untuk text, numerical features, dan genre encoding"
)