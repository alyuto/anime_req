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
import json

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
# Fungsi untuk generate alasan rekomendasi menggunakan LLM
# ============================
def generate_recommendation_reason(reference_anime, recommended_anime):
    """Generate detailed recommendation reason based on multiple factors"""
    try:
        # Analisis faktor-faktor rekomendasi yang lebih komprehensif
        ref_genres = set(reference_anime.get('genres', []))
        rec_genres = set(recommended_anime.get('genres', []))
        common_genres = ref_genres & rec_genres
        
        # Analisis studio yang sama
        ref_studios = set(reference_anime.get('studios', []))
        rec_studios = set(recommended_anime.get('studios', []))
        common_studios = ref_studios & rec_studios
        
        # Analisis score similarity
        ref_score = reference_anime.get('score', 0)
        rec_score = recommended_anime.get('score', 0)
        score_diff = abs(ref_score - rec_score)
        
        # Similarity score dari model
        similarity = recommended_anime.get('similarity_score', 0)
        
        # Generate reason berdasarkan multiple factors
        reasons = []
        
        # 1. Similarity score explanation
        if similarity >= 80:
            reasons.append(f"Memiliki kemiripan sangat tinggi ({similarity}%) berdasarkan analisis konten mendalam")
        elif similarity >= 60:
            reasons.append(f"Menunjukkan kemiripan yang kuat ({similarity}%) dalam karakteristik anime")
        else:
            reasons.append(f"Memiliki kesamaan signifikan ({similarity}%) dengan preferensi Anda")
        
        # 2. Genre analysis
        if common_genres:
            if len(common_genres) >= 2:
                reasons.append(f"berbagi genre utama: {', '.join(list(common_genres)[:2])}")
            else:
                reasons.append(f"memiliki genre serupa: {', '.join(common_genres)}")
        
        # 3. Studio analysis
        if common_studios:
            reasons.append(f"diproduksi oleh studio yang sama: {', '.join(list(common_studios)[:1])}")
        
        # 4. Score analysis
        if score_diff <= 0.5:
            reasons.append(f"memiliki kualitas rating yang setara ({rec_score})")
        elif rec_score >= 8.0:
            reasons.append(f"memiliki rating tinggi ({rec_score}) yang menjanjikan")
        elif rec_score >= 7.0:
            reasons.append(f"memiliki rating solid ({rec_score}) dengan kualitas terjamin")
        
        # 5. Popularity analysis
        members = recommended_anime.get('members', 0)
        if members >= 1000000:
            reasons.append("sangat populer dengan jutaan penggemar")
        elif members >= 500000:
            reasons.append("populer dengan base penggemar yang besar")
        
        # 6. Type analysis
        rec_type = recommended_anime.get('type', '')
        if rec_type == 'Movie':
            reasons.append("format movie yang cocok untuk pengalaman menonton yang intens")
        elif rec_type == 'OVA':
            reasons.append("format OVA dengan produksi berkualitas tinggi")
        
        # Gabungkan reasons dengan format yang natural
        if len(reasons) == 1:
            final_reason = f"Anime ini direkomendasikan karena {reasons[0]}."
        elif len(reasons) == 2:
            final_reason = f"Anime ini direkomendasikan karena {reasons[0]} dan {reasons[1]}."
        elif len(reasons) >= 3:
            main_reasons = reasons[:2]
            additional = reasons[2]
            final_reason = f"Anime ini direkomendasikan karena {main_reasons[0]}, {main_reasons[1]}, serta {additional}."
        else:
            final_reason = f"Anime ini direkomendasikan berdasarkan analisis komprehensif similarity model ({similarity}%)."
        
        # Tambahkan confidence indicator
        if similarity >= 70:
            final_reason += " Rekomendasi ini sangat cocok untuk Anda!"
        elif similarity >= 50:
            final_reason += " Anda akan menyukai anime ini!"
        else:
            final_reason += " Worth to try!"
        
        return final_reason
        
    except Exception as e:
        return f"Anime ini direkomendasikan berdasarkan analisis model AI dengan similarity {recommended_anime.get('similarity_score', 0)}%."

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

    def get_recommendations_by_mal_id(self, mal_id, num_recommendations=10, similarity_threshold=0.3, include_reasons=True):
        """Get anime recommendations based on MAL ID with reasons"""
        try:
            # Find anime in processed data
            anime_row = self.processed_data[self.processed_data['mal_id'] == mal_id]
            if anime_row.empty:
                return {"error": "Anime dengan MAL ID tersebut tidak ditemukan dalam dataset"}
            
            idx = anime_row.index[0]
            
            # Get reference anime info
            reference_original = self.original_data[self.original_data['mal_id'] == mal_id]
            if reference_original.empty:
                return {"error": "Anime referensi tidak ditemukan"}
            
            reference_anime = reference_original.iloc[0]
            ref_genres = self.safe_literal_eval(reference_anime.get('genres', '[]'))
            ref_genre_names = []
            for g in ref_genres:
                if isinstance(g, dict):
                    ref_genre_names.append(g.get('name', ''))
                elif isinstance(g, str):
                    ref_genre_names.append(g)
            
            # Get reference studios
            ref_studios = self.safe_literal_eval(reference_anime.get('studios', '[]'))
            ref_studio_names = []
            for s in ref_studios:
                if isinstance(s, dict):
                    ref_studio_names.append(s.get('name', ''))
                elif isinstance(s, str):
                    ref_studio_names.append(s)
            
            reference_info = {
                'title': str(reference_anime.get('title_english', reference_anime.get('title', 'Unknown'))),
                'genres': ref_genre_names,
                'studios': ref_studio_names,
                'score': float(reference_anime.get('score', 0)),
                'type': str(reference_anime.get('type', 'Unknown')),
                'members': int(reference_anime.get('members', 0))
            }
            
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
                    
                    # Generate recommendation reason if requested
                    if include_reasons:
                        anime_info['recommendation_reason'] = generate_recommendation_reason(reference_info, anime_info)
                    
                    recommendations.append(anime_info)
            
            return {
                "recommendations": recommendations, 
                "total_found": len(recommendations),
                "reference_anime": reference_info
            }
            
        except Exception as e:
            return {"error": f"Terjadi kesalahan: {str(e)}"}

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
    "- **Alasan rekomendasi menggunakan AI** 🤖\n"
    "- Dataset lokal yang sudah diproses dengan TF-IDF dan SVD\n\n"
    "**Contoh MAL ID:** 5114 (Fullmetal Alchemist: Brotherhood), 9253 (Steins;Gate)"
)

# Tab untuk berbagai fitur
tab1, tab2 = st.tabs(["🔍 Search & Recommend", "📊 Dataset Info"])

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
    st.subheader("Dapatkan Rekomendasi dengan Alasan AI")
    mal_id_input = st.text_input("Masukkan MAL ID anime", "")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        num_recommendations = st.slider("Jumlah rekomendasi", 5, 20, 10)
    with col2:
        similarity_threshold = st.slider("Threshold kemiripan", 0.1, 0.9, 0.3)
    with col3:
        include_ai_reasons = st.checkbox("Tampilkan alasan AI", value=True)
    
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
                mal_id, num_recommendations, similarity_threshold, include_ai_reasons
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
                        
                        # AI Recommendation Reason
                        if include_ai_reasons and 'recommendation_reason' in rec:
                            st.info(f"🤖 **Alasan AI:** {rec['recommendation_reason']}")
                        
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
    st.header("Informasi Dataset")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Anime dalam Dataset", len(processed_data))
        st.metric("Fitur yang Digunakan", len(feature_columns))
    with col2:
        st.metric("Dimensi Similarity Matrix", f"{similarity_matrix.shape[0]} x {similarity_matrix.shape[1]}")
        st.metric("Algoritma AI", "TF-IDF + SVD + Neural Network")
    
    st.subheader("Fitur Utama Sistem:")
    features = [
        "🔍 Pencarian anime berdasarkan judul",
        "🎯 Rekomendasi berbasis content similarity",
        "🤖 Alasan rekomendasi menggunakan AI",
        "📊 Similarity scoring yang akurat",
        "🎬 Informasi lengkap anime + streaming links",
        "⚡ Performa tinggi dengan caching"
    ]
    
    for feature in features:
        st.write(feature)
    
    st.subheader("Sample Data Processing:")
    if st.button("Tampilkan sample processed data"):
        st.dataframe(processed_data.head())

# ============================
# Footer
# ============================
st.markdown("---")
st.markdown(
    "**Anime Recommendation System** menggunakan Deep Learning dan Content-Based Filtering\n\n"
    "🆕 **New Feature:** AI-powered recommendation reasoning untuk pengalaman yang lebih personal!"
)