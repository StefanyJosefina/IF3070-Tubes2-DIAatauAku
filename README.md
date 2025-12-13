# IF3070-Tubes2-DIAatauAku

## 1. Deskripsi Singkat

Repositori ini merupakan implementasi Tugas Besar 2 mata kuliah **IF3070 – Artificial Intelligence**  dari kelompok **“DIA atau Aku”**. 

Proyek ini berfokus pada implementasi algoritma Supervised Machine Learning dari awal (scratch). Scikit-learn digunakan sebagai baseline pembanding terhadap implementasi algoritma dari scratch.

Tujuan utama proyek ini adalah membangun dan menganalisis kinerja model pembelajaran mesin tanpa bergantung sepenuhnya pada pustaka siap pakai untuk logika inti algoritma. 

Untuk menghasilkan prediksi yang berkualitas, dilakukan beberapa tahapan berikut: **Data Cleaning, Data Transformation, Feature Selection, dan Dimensionality Reduction**.

Algoritma yang diimplementasikan mencakup: 
- **Decision Tree Learning (DTL)**: Mendukung fitur numerik dan kategorikal, serta kriteria impurity (Gini/Entropy).
- **K-Nearest Neighbors (KNN)**: Mendukung berbagai metrik jarak (Euclidean, Manhattan, Minkowski) dan pembobotan (uniform/distance).

Struktur program dikembangkan menggunakan bahasa Python dengan pendekatan Object-Oriented Programming (OOP) yang kompatibel dengan antarmuka scikit-learn (`BaseEstimator`, `ClassifierMixin`).

## 2. Struktur Direktori

```
IF3070-Tubes2-DIAatauAku/
│
├── src/
│   ├── dtl.py               # Implementasi algoritma Decision Tree (ID3/CART like)
│   ├── knn.py               # Implementasi algoritma K-Nearest Neighbors
│   └── .gitignore           # Konfigurasi git ignore
│
├── data/
│   ├── train. csv            # Data latih
│   ├── test.csv              # Data uji
│   └── sample_submission.csv # Format submisi
│
├── notebook.ipynb           # Jupyter Notebook untuk eksperimen dan visualisasi
├── requirements.txt         # Daftar dependensi Python
└── README.md                # Dokumentasi proyek
```

## 3. Cara Menjalankan Program

### 3.1 Clone Repository

```bash
git clone https://github.com/StefanyJosefina/IF3070-Tubes2-DIAatauAku.git
cd IF3070-Tubes2-DIAatauAku
```

### 3.2 Instalasi Dependensi

Disarankan untuk menggunakan virtual environment.  Instal dependensi dengan perintah:

```bash
python -m venv .venv
.venv\Scripts\activate 
pip install -r requirements.txt
```

### 3.3 Menjalankan Eksperimen

Eksperimen utama dan pengujian model dapat dilakukan melalui Jupyter Notebook. Jalankan perintah berikut: 

```bash
jupyter notebook notebook.ipynb
```

Atau Anda dapat mengimpor kelas secara langsung dalam skrip Python:

```python
from src.dtl import DTL
from src.knn import KNN

# Contoh penggunaan
model = DTL(max_depth=5, criterion="entropy")
model.fit(X_train, y_train)
prediction = model.predict(X_test)
```

## 4. Spesifikasi Teknis

- **Bahasa Pemrograman**: Python
- **Library Pendukung**: numpy, pandas, scikit-learn

### Algoritma yang Diimplementasikan:

#### A. Decision Tree Learning (DTL)

**Basis**: Class `DTL` (turunan `BaseEstimator`, `ClassifierMixin`).

**Fitur Utama**:
- Penanganan atribut numerik (dengan thresholding otomatis) dan kategorikal.
- Dukungan kriteria split:  Gini Index dan Entropy (Information Gain).
- Hyperparameters: `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.
- Optimasi quantile-based thresholding untuk fitur numerik dengan kardinalitas tinggi. 

#### B. K-Nearest Neighbors (KNN)

**Basis**: Class `KNN` (turunan `BaseEstimator`, `ClassifierMixin`).

**Fitur Utama**:
- Dukungan metrik jarak: Manhattan (p=1), Euclidean (p=2), dan Minkowski.
- Metode pembobotan: Uniform (suara mayoritas) dan Distance (pembobotan invers jarak).
- Implementasi fungsi `predict` dan `predict_proba`.

## 5. Pembagian Tugas Kelompok

| NIM | Nama | Tugas / Kontribusi |
|-----|------|-------------------|
| 18223116 | Stefany Josefina Santono | Implementasi DTL, Menyusun Laporan |
| 18223125 | Matilda Angelina Sumaryo | Implementasi KKN, Menyusun Laporan |
| 18223139 | Andi Syaichul Mubaraq | Preprocessing Data, Menyusun Laporan |