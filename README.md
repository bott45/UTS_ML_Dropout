# UTS_ML_Dropout
# Prediksi Risiko Dropout Mahasiswa (UTS Machine Learning, MITI.302)

Klasifikasi biner untuk memprediksi mahasiswa yang berisiko dropout, memakai dataset
*Predict Students' Dropout and Academic Success* (UCI, 4.424 mahasiswa, 36 fitur).

## Anggota
Muhammad Farid Abdillah(251552010008)
Fabian Alfarizi(251552010039)

## Struktur
- `data/raw/` data asli, `data/processed/` hasil pembagian train (80%) dan test (20%)
- `notebooks/` 01_eda, 02_praproses, 03_model_cv
- `src/pipeline.py` praproses (one-hot, penskalaan) dalam Pipeline
- `results/` hasil nested CV dan hasil uji

## Cara menjalankan
pip install -r requirements.txt, lalu jalankan notebook berurutan 01, 02, 03.

## Desain evaluasi
Target biner (Dropout vs bukan). Dua skenario: tanpa dan dengan fitur akademik semester.
Lima model (Baseline, Decision Tree, Random Forest, SVM, kNN), nested CV, data uji dibuka sekali.

## Hasil (data uji, Random Forest)
| Skenario | F1 | Recall | ROC-AUC |
|---|---|---|---|
| Tanpa akademik | 0,650 | 0,701 | 0,816 |
| Dengan akademik | 0,795 | 0,806 | 0,921 |

## Rujukan
Realinho dkk., "Predict Students' Dropout and Academic Success," UCI ML Repository, 2021.