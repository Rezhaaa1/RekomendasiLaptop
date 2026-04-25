import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Sistem Rekomendasi Laptop",
    page_icon="💻",
    layout="wide"
)

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_laptop.csv")
    df = df.drop_duplicates()
    return df

df_original = load_data()

features = [
    "cpu_score",
    "gpu_score",
    "ram_gb",
    "storage_gb",
    "screen_size_inch",
    "resolution_pixel"
]

df_original = df_original.dropna(
    subset=features + ["price"]
).reset_index(drop=True)

# ======================================================
# NORMALISASI
# ======================================================
df_scaled = df_original.copy()
scaler = MinMaxScaler()
df_scaled[features] = scaler.fit_transform(df_scaled[features])

# ======================================================
# FORMAT RUPIAH
# ======================================================
def format_rupiah(x):
    return f"Rp {int(x):,}".replace(",", ".")

# ======================================================
# FILTER DATA
# ======================================================
def filter_data(df_ori, df_scale, min_budget, max_budget, min_ram, min_storage, brand):

    mask = (
        (df_ori["price"] >= min_budget) &
        (df_ori["price"] <= max_budget) &
        (df_ori["ram_gb"] >= min_ram) &
        (df_ori["storage_gb"] >= min_storage)
    )

    if brand != "Semua":
        mask &= (df_ori["brand"] == brand)

    return df_ori[mask].copy(), df_scale[mask].copy()

# ======================================================
# KNN REKOMENDASI
# ======================================================
def knn_recommendation(df_ori, df_scale, top_k, user_input):

    if df_scale.empty:
        return pd.DataFrame()

    weights = np.array([0.35, 0.25, 0.2, 0.1, 0.05, 0.05])

    user_df = pd.DataFrame([user_input], columns=features)
    user_scaled = scaler.transform(user_df)

    weighted_data = df_scale[features] * weights
    weighted_user = user_scaled * weights

    knn = NearestNeighbors(
        n_neighbors=min(top_k, len(df_scale)),
        metric="euclidean"
    )

    knn.fit(weighted_data)

    distances, indices = knn.kneighbors(weighted_user)

    result = df_ori.iloc[indices[0]].copy()
    result["similarity_score"] = 1 / (1 + distances[0])

    return result.sort_values(by="similarity_score", ascending=False)

# ======================================================
# 🔥 SIDEBAR MINIMALIS & PROFESIONAL
# ======================================================
st.sidebar.markdown("""
<div style="
    background: linear-gradient(135deg, #1f77b4, #4CAF50);
    padding:15px;
    border-radius:12px;
    text-align:center;
    color:white;
    margin-bottom:20px;
">
    <h2 style="margin-bottom:5px;">💻</h2>
    <p style="font-size:12px;">Sistem Rekomendasi Laptop</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Menu Navigasi")

page = st.sidebar.radio(
    "",
    ["📊 Dashboard", "🔍 Rekomendasi Laptop"]
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<p style="font-size:12px; text-align:center; color:gray;">
Dikembangkan untuk penelitian skripsi<br>
Sistem Rekomendasi berbasis KNN
</p>
""", unsafe_allow_html=True)

if page == "📊 Dashboard":

    # ======================================================
    # 🔥 HERO HEADER FUTURISTIK
    # ======================================================
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0f2027, #2c5364);
        padding:30px;
        border-radius:15px;
        color:white;
        text-align:center;
        margin-bottom:20px;
    ">
        <h1>💻 Sistem Rekomendasi Laptop</h1>
        <p style="font-size:16px;">
        Aplikasi ini membantu pengguna menemukan laptop terbaik 
        berdasarkan kebutuhan dan preferensi menggunakan metode K-Nearest Neighbor (KNN).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # 🔥 DESKRIPSI APLIKASI
    # ======================================================
    st.markdown("""
    ### 💰 Informasi Harga
    Seluruh data harga laptop yang digunakan dalam sistem ini merupakan **harga laptop baru (bukan second)** 
    yang diambil dari kisaran harga pasar di website official dari merk laptop nya langsung. Hal ini bertujuan agar rekomendasi yang diberikan tetap relevan 
    dan sesuai dengan kondisi pasar saat ini.
    """)

    st.markdown("---")

    # ======================================================
    # 🔥 STATISTIK DATA (CARD STYLE)
    # ======================================================
    st.subheader("📊 Statistik Dataset")

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"""
    <div style="background:#111; padding:20px; border-radius:12px; color:white; text-align:center;">
        <h2>{len(df_original)}</h2>
        <p>Total Laptop</p>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div style="background:#1f77b4; padding:20px; border-radius:12px; color:white; text-align:center;">
        <h2>{format_rupiah(df_original["price"].min())}</h2>
        <p>Harga Termurah</p>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div style="background:#4CAF50; padding:20px; border-radius:12px; color:white; text-align:center;">
        <h2>{format_rupiah(df_original["price"].max())}</h2>
        <p>Harga Termahal</p>
    </div>
    """, unsafe_allow_html=True)

    col4.markdown(f"""
    <div style="background:#ff9800; padding:20px; border-radius:12px; color:white; text-align:center;">
        <h2>{df_original["ram_gb"].max()} GB</h2>
        <p>RAM Maksimal</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ======================================================
    # 🔥 HIGHLIGHT FITUR
    # ======================================================
    st.subheader("🚀 Fitur Utama")

    colA, colB, colC = st.columns(3)

    colA.markdown("""
    <div style="padding:15px; border-radius:10px; background:#f5f5f5;">
    🔍 <b>Rekomendasi Cerdas</b><br>
    Menggunakan algoritma KNN untuk memberikan hasil terbaik.
    </div>
    """, unsafe_allow_html=True)

    colB.markdown("""
    <div style="padding:15px; border-radius:10px; background:#f5f5f5;">
    ⚡ <b>Proses Cepat</b><br>
    Hasil rekomendasi ditampilkan secara real-time.
    </div>
    """, unsafe_allow_html=True)

    colC.markdown("""
    <div style="padding:15px; border-radius:10px; background:#f5f5f5;">
    🎯 <b>Akurat & Relevan</b><br>
    Berdasarkan kebutuhan spesifik pengguna.
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# HALAMAN REKOMENDASI
# ======================================================
else:

    st.title("🔍 Sistem Rekomendasi Laptop")
    st.write("Masukkan kebutuhan Anda untuk mendapatkan rekomendasi terbaik")

    col1, col2 = st.columns(2)

    with col1:
        min_budget = st.number_input(
            "💰 Budget Minimum",
            1_000_000, 50_000_000, 5_000_000, step=500_000
        )
    with col2:
        max_budget = st.number_input(
            "💰 Budget Maksimum",
            min_budget, 100_000_000, 15_000_000, step=500_000
        )

    col3, col4 = st.columns(2)

    with col3:
        min_ram = st.selectbox("🧠 Minimal RAM (GB)", [4, 8, 16, 32])
    with col4:
        min_storage = st.selectbox("💾 Minimal Storage (GB)", [256, 512, 1024])

    col5, col6 = st.columns(2)

    with col5:
        brand = st.selectbox(
            "🏷️ Pilih Brand",
            ["Semua"] + sorted(df_original["brand"].dropna().unique())
        )
    with col6:
        top_k = st.selectbox("📊 Jumlah Rekomendasi", [5, 10])

    if st.button("🔍 Cari Rekomendasi", use_container_width=True):

        if min_ram <= 8:
            cpu_pref = 6500
            gpu_pref = 2000
        elif min_ram == 16:
            cpu_pref = 8500
            gpu_pref = 4000
        else:
            cpu_pref = 10000
            gpu_pref = 6000

        user_input = {
            "cpu_score": cpu_pref,
            "gpu_score": gpu_pref,
            "ram_gb": min_ram,
            "storage_gb": min_storage,
            "screen_size_inch": 14,
            "resolution_pixel": 2000000
        }

        df_f_ori, df_f_scale = filter_data(
            df_original, df_scaled,
            min_budget, max_budget,
            min_ram, min_storage,
            brand
        )

        result = knn_recommendation(
            df_f_ori,
            df_f_scale,
            top_k,
            user_input
        )

        if result.empty:
            st.error("❌ Tidak ada laptop yang sesuai dengan kriteria.")
        else:
            st.success("✅ Rekomendasi ditemukan")

            result["price"] = result["price"].apply(format_rupiah)

            # Ranking
            result = result.reset_index(drop=True)
            result.insert(0, "No", result.index + 1)

            st.subheader("🏆 Hasil Rekomendasi")

            # ======================================================
            # HERO (BEST)
            # ======================================================
            top1 = result.iloc[0]

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1f77b4, #4CAF50);
                padding:20px;
                border-radius:15px;
                color:white;
                margin-bottom:20px;
            ">
                <h2>🥇 Rekomendasi Terbaik</h2>
                <h3>{top1['model_name']}</h3>
                <p><b>Brand:</b> {top1['brand']}</p>
                <p><b>Harga:</b> {top1['price']}</p>
                <p>⚡ CPU: {top1['CPU']} | 🎮 GPU: {top1['GPU']}</p>
                <p>🧠 RAM: {top1['ram_gb']} GB | 💾 Storage: {top1['storage_gb']} GB</p>
            </div>
            """, unsafe_allow_html=True)

            # ======================================================
            # GRID CARD
            # ======================================================
            cols = st.columns(3)

            for i, row in result.iterrows():
                with cols[i % 3]:

                    if row["No"] == 1:
                        badge = "🥇"
                    elif row["No"] == 2:
                        badge = "🥈"
                    elif row["No"] == 3:
                        badge = "🥉"
                    else:
                        badge = f"{row['No']}."

                    st.markdown(f"""
                    <div style="
                        border:1px solid #eee;
                        border-radius:15px;
                        padding:15px;
                        margin-bottom:15px;
                        background-color:#ffffff;
                        box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
                    ">

                    <h4>{badge} {row['model_name']}</h4>
                    <p><b>{row['brand']}</b></p>
                    <p style="color:green;"><b>{row['price']}</b></p>

                    <hr>

                    <p>⚙️ {row['CPU']}</p>
                    <p>🎮 {row['GPU']}</p>
                    <p>🧠 {row['ram_gb']} GB | 💾 {row['storage_gb']} GB</p>
                    <p>🖥️ {row['screen_size_inch']} inch</p>

                    </div>
                    """, unsafe_allow_html=True)

                  