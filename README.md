# Sesi 4: Machine Learning & Tren Masa Depan AI
**Pipeline *Hybrid Retrieval-Augmented Generation* (RAG) untuk AI Tutor Bebas Halusinasi**

## 📖 Deskripsi Studi Kasus
Dalam penerapan *Artificial Intelligence* (AI) berbasis *Large Language Model* (LLM) di sektor pendidikan atau pemerintahan, masalah terbesar yang sering muncul adalah **halusinasi**—kondisi di mana AI memberikan jawaban yang terdengar meyakinkan namun faktanya salah atau mengarang bebas karena tidak memiliki dasar referensi yang spesifik.

Studi kasus ini mendemonstrasikan perancangan arsitektur mesin pencari dokumen hibrida (*Hybrid Retrieval*) yang menjadi jantung dari sistem RAG. Sebelum pertanyaan mahasiswa diteruskan ke AI untuk dijawab, sistem ini akan secara otomatis membongkar modul-modul akademik (*Knowledge Base*) guna menemukan paragraf konteks yang paling relevan. Pendekatan hibrida mengombinasikan pencarian leksikal (kecocokan kata kunci/TF-IDF) untuk presisi istilah teknis, dan pencarian semantik (makna kalimat) untuk memahami konteks pertanyaan.

## 🎯 Tujuan Eksekusi
1. **Ekstraksi Pengetahuan (Knowledge Extraction):** Membangun basis data vektor sederhana dari potongan dokumen/modul akademik.
2. **Pencarian Leksikal (Lexical Retrieval):** Mengimplementasikan algoritma berbasis frekuensi kata (seperti TF-IDF/BM25) untuk menangkap kecocokan istilah spesifik (misal: "Hukum Gestalt").
3. **Penyediaan Konteks yang Kokoh (Fact Grounding):** Menyediakan konteks referensi yang mutlak bagi LLM agar jawaban yang dihasilkan bersumber murni dari literatur resmi, bukan dari tebakan model.

## 💻 Implementasi Kode (Python - Scikit-Learn untuk RAG Retrieval Pipeline)
Berikut adalah skrip purwarupa mesin pencari konteks yang dapat Anda jalankan melalui file `hybrid_rag_retrieval.py`.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("=== PIPELINE RETRIEVAL RAG UNTUK AI TUTOR ===\n")

# 1. Membangun Basis Pengetahuan (Knowledge Base) dari Modul Akademik
database_modul = [
    "Machine Learning adalah cabang AI yang fokus pada penggunaan data dan algoritma untuk meniru cara manusia belajar.",
    "Hukum Gestalt dalam desain antarmuka menyatakan bahwa manusia secara visual mengelompokkan elemen yang berdekatan atau serupa.",
    "Overfitting terjadi ketika model Machine Learning terlalu menghafal data latih sehingga gagal memprediksi data baru.",
    "Pemrosesan citra dengan High Performance Computing memanfaatkan GPU untuk mengeksekusi matriks secara paralel.",
    "Kelemahan sistem peringatan dini konvensional adalah tingginya extraneous load akibat antarmuka yang penuh chartjunk."
]

# 2. Inisialisasi Mesin Pencari Leksikal (TF-IDF Vectorizer)
# Pada skala industri, ini merepresentasikan fungsi BM25
vectorizer = TfidfVectorizer()
vektor_dokumen = vectorizer.fit_transform(database_modul)

def cari_konteks_relevan(pertanyaan, top_k=1):
    """
    Fungsi untuk mencari paragraf paling relevan di dalam database 
    berdasarkan pertanyaan pengguna.
    """
    # Mengubah pertanyaan menjadi vektor matematis
    vektor_pertanyaan = vectorizer.transform([pertanyaan])
    
    # Menghitung kemiripan (Cosine Similarity) antara pertanyaan dan seluruh dokumen
    skor_kemiripan = cosine_similarity(vektor_pertanyaan, vektor_dokumen).flatten()
    
    # Mengurutkan hasil dari skor tertinggi ke terendah
    indeks_terbaik = np.argsort(skor_kemiripan)[::-1][:top_k]
    
    hasil = []
    for idx in indeks_terbaik:
        if skor_kemiripan[idx] > 0: # Pastikan ada relevansi
            hasil.append((skor_kemiripan[idx], database_modul[idx]))
            
    return hasil

# 3. Skenario Pengujian (Simulasi Interaksi Mahasiswa)
daftar_pertanyaan = [
    "Apa penyebab utama model gagal memprediksi data baru?",
    "Jelaskan mengenai Hukum Gestalt dalam UI/UX!",
    "Bagaimana cara memasak rendang yang enak?" # Pertanyaan di luar konteks (Out of Domain)
]

print("--- SIMULASI PENCARIAN KONTEKS (FACT GROUNDING) ---")
for q in daftar_pertanyaan:
    print(f"\n🙋‍♂️ Pertanyaan Mahasiswa : '{q}'")
    konteks_ditemukan = cari_konteks_relevan(q)
    
    print("🤖 Sistem Retrieval RAG :")
    if konteks_ditemukan:
        skor, teks = konteks_ditemukan[0]
        print(f"   [✓] Konteks Ditemukan (Skor Relevansi: {skor:.2f})")
        print(f"   [>] '{teks}'")
        print("   [INFO] Konteks ini akan dikirim ke LLM (IndoBERT/GPT) untuk dirangkum menjadi jawaban.")
    else:
        print("   [X] Peringatan: Tidak ada konteks yang relevan di dalam modul akademik!")
        print("   [INFO] AI Tutor akan diinstruksikan untuk menjawab: 'Maaf, materi tersebut tidak ada dalam silabus.' (Mencegah Halusinasi)")
