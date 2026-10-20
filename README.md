# Arsitektur Hybrid RAG untuk AI Tutor: Mencegah Halusinasi pada Domain Edukasi

## Deskripsi

Proyek ini merupakan purwarupa arsitektur kecerdasan buatan terapan (*Applied AI*) yang berfokus pada pencegahan halusinasi *Large Language Model* (LLM) di sektor pendidikan. 
Proyek ini mengimplementasikan prinsip dasar *Retrieval-Augmented Generation* (RAG) secara hibrida, seperti yang dibahas dalam materi praktisi AI terkini mengenai pentingnya penyediaan konteks atau *fact grounding* sebelum membiarkan model AI men-*generate* jawaban.

## Tujuan

Tujuan proyek ini adalah:
1. Memecahkan celah kelemahan (*gap*) LLM konvensional yang sering kali berhalusinasi atau memberikan jawaban umum di luar silabus kampus.
2. Membangun basis pengetahuan (*Knowledge Base*) dari kumpulan modul akademik.
3. Mengimplementasikan algoritma pencarian leksikal (berbasis TF-IDF/BM25) untuk mengukur relevansi dokumen secara matematis (*Cosine Similarity*).
4. Mengevaluasi keandalan sistem dalam membedakan pertanyaan sesuai domain (*In-Domain*) versus pertanyaan di luar materi (*Out-of-Domain*).

## Studi Kasus

Fakultas Teknik Universitas Bengkulu berencana mengimplementasikan *AI Tutor* untuk membantu mahasiswa belajar 24/7. Namun, sistem uji coba awal gagal karena AI sering menjawab pertanyaan teknis (seperti pemrograman atau matematika) dengan konsep yang salah, atau justru meladeni pertanyaan mahasiswa yang tidak relevan (seperti resep masakan).
Solusinya adalah membangun *pipeline Retrieval RAG*. RAG berfungsi sebagai "pustakawan" digital. Saat mahasiswa bertanya, sistem tidak langsung memanggil ChatGPT/LLM. RAG akan terlebih dahulu menggeledah *database* modul resmi untuk mencari paragraf yang relevan. Jika konteks ditemukan, barulah LLM bertugas merangkumnya. Jika tidak ada di modul, sistem secara tegas akan menolak menjawab, sehingga risiko halusinasi dapat ditekan hingga 0%.

## Dataset

Dataset yang digunakan merupakan kumpulan teks sintetis yang merepresentasikan 25 paragraf modul akademik Informatika (mulai dari materi Jaringan Komputer, Rekayasa Perangkat Lunak, hingga Sistem Cerdas). Dataset ini di-*generate* menggunakan algoritma *list comprehension* untuk menyimulasikan basis pengetahuan fakultas.

## Metode

Tahapan utama yang dilakukan:
1. *Corpus Initialization* (Pembuatan kumpulan modul pengetahuan).
2. *Vectorization* (Transformasi dokumen teks menjadi matriks angka 2D menggunakan algoritma *Term Frequency-Inverse Document Frequency* / TF-IDF).
3. *Similarity Scoring* (Penghitungan jarak metrik antarkata menggunakan *Cosine Similarity*).
4. *Threshold Tuning* (Menetapkan batas skor minimal/ambang batas agar sistem berani menolak pertanyaan di luar konteks).

## Evaluasi

Performa sistem *retrieval* dievaluasi menggunakan:
*   **Relevance Score (0.0 - 1.0):** Tingkat kemiripan dokumen dengan pertanyaan.
*   **Domain Segregation Accuracy:** Kemampuan sistem menolak 100% pertanyaan yang sengaja disisipkan dari luar domain (misal: pertanyaan tentang politik atau memasak).
Berbeda dengan *Machine Learning* klasik yang mengevaluasi prediksi (*Accuracy/Recall*), evaluasi sistem RAG sangat bergantung pada kemampuan mesin pencarinya ( *Retriever*) untuk menemukan dokumen rujukan yang mutlak benar.

## Teknologi

*   Python
*   Google Colab / Jupyter Notebook (`.ipynb`)
*   Scikit-Learn (`TfidfVectorizer`, `cosine_similarity`)
*   NumPy
*   Pandas

## Cara Menjalankan

1. Proyek dapat dijalankan menggunakan Google Colab.
2. Buka file `rag_hybrid_retrieval.ipynb` menggunakan Google Colab.
3. Jalankan *notebook* dari awal hingga akhir.
4. Pada *cell* terakhir, Anda akan melihat matriks evaluasi dan simulasi tanya-jawab *AI Tutor* yang merespons pertanyaan *in-domain* maupun *out-of-domain* secara otomatis.


## Kesimpulan
Proyek ini membuktikan bahwa masa depan Generative AI bukanlah sekadar menggunakan LLM terbesar, melainkan mengontrol LLM tersebut dengan arsitektur data engineering yang solid.
Dengan menerapkan pipeline RAG, kita menggeser peran AI dari sekadar "penghasil teks yang fasih berhalusinasi" menjadi "asisten akademik yang berbasis pada fakta". Implementasi skor ambang batas (threshold) memastikan integritas akademik tetap terjaga, memisahkan secara tegas mana materi resmi kampus dan mana pertanyaan di luar silabus.

## Struktur Repository

```text
sesi-4-ml/
├── rag_hybrid_retrieval.ipynb
├── README.md
└── requirements.txt
