# Indonesian Play Store Sentiment Analysis 📱💬

[![Dicoding - Bintang 5](https://img.shields.io/badge/Dicoding_Submission-Bintang_5-gold.svg)](https://www.dicoding.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![IndoBERT](https://img.shields.io/badge/Model-IndoBERT-orange.svg)](https://huggingface.co/indobenchmark/indobert-base-p2)

Proyek ini merupakan sistem **Analisis Sentimen Klasifikasi Multikelas** (Positif, Netral, Negatif) pada bahasa Indonesia *informal* (bahasa gaul, slang, typo) yang diekstraksi langsung dari ulasan Google Play Store. 

Proyek ini dibangun untuk memenuhi syarat kelulusan **Bintang 5** pada kelas Machine Learning di Dicoding Academy dengan pencapaian akurasi model lebih dari **92%**.

## 📌 Fitur Utama
- **Automated Data Scraping:** Menggunakan `google-play-scraper` untuk menambang belasan ribu ulasan unik dari aplikasi populer (Tokopedia, Gojek, DANA).
- **Fast Data Processing:** Memanfaatkan `Polars` untuk pra-pemrosesan data berskala besar secara kilat.
- **Deep Data Cleansing:** Membersihkan *noise* ambigu (bintang 2 dan 4) untuk memperkuat polaritas data, serta mengaplikasikan teknik *Early Oversampling* untuk mengatasi ketimpangan kelas (*Imbalanced Dataset*).
- **Multi-Model Experiments:** Melakukan eksperimen pada 3 arsitektur pembelajaran mesin (Machine Learning & Deep Learning) yang berbeda secara komprehensif.

## 🔬 Arsitektur Model & Perbandingan Akurasi
Proyek ini bereksperimen dengan 3 pendekatan mulai dari algoritma klasik hingga *State-of-the-Art Transformer*:

| Skema | Metode Ekstraksi | Algoritma Pelatihan | Train Acc | Test Acc |
|:---:|:---|:---|:---:|:---:|
| **1** | TF-IDF (N-Grams) | Support Vector Machine (SVM) | 95.03% | 89.52% |
| **2** | Word2Vec (128d) | Bidirectional GRU (Bi-GRU) | 95.76% | 89.63% |
| **3** | AutoTokenizer | **IndoBERT Base P2** (Fine-tuning) | **99.08%** | **92.87%** |

*Catatan: Model IndoBERT (Transfer Learning) berhasil menyentuh standar emas >92% test accuracy.*

## 🚀 Cara Menjalankan

### 1. Instalasi Dependencies
Pastikan Anda memiliki Python 3.10+ dan *environment* yang bersih.
```bash
pip install -r requirements.txt
```

### 2. Scraping Data Baru (Opsional)
Jika Anda ingin menambang data terbaru dari Play Store secara langsung:
```bash
python scraping.py
# atau jalankan scraping.ipynb di Jupyter/Colab
```

### 3. Pelatihan & Evaluasi Model
Untuk melakukan *training* ulang pada ketiga model (memerlukan GPU/NVIDIA CUDA untuk IndoBERT):
```bash
python training.py
# atau jalankan training.ipynb di Google Colab
```

## 🧠 Contoh Inferensi (Test Model)
Model IndoBERT dapat langsung menebak sentimen dari kalimat bebas:
```python
teks = "Aplikasi ini sangat membantu pekerjaan saya sehari-hari, top markotop!"
# Output: Positif

teks = "Sangat mengecewakan, fitur transfer error terus dan CS tidak responsif"
# Output: Negatif
```

## 🏆 Kriteria Kelulusan (Dicoding)
- [x] Scraping mandiri minimal 3.000 sampel (Total: 13.000+ sampel).
- [x] 3 Kelas klasifikasi (Positif, Netral, Negatif).
- [x] Uji coba 3 variasi algoritma dan *feature extraction* yang berbeda.
- [x] Algoritma Deep Learning.
- [x] Akurasi pengujian **>92%**.

