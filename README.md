# UTS_ML_Dropout
# Prediksi Risiko Dropout Mahasiswa (UTS Machine Learning, MITI.302)

Klasifikasi biner untuk memprediksi mahasiswa yang berisiko dropout, memakai dataset
*Predict Students' Dropout and Academic Success* (UCI, 4.424 mahasiswa, 36 fitur).

## Anggota
Muhammad Farid Abdillah(251552010008)
Fabian Alfarizi(251552010039)

## Rujukan penelitian
Dataset dan rancangan mengacu pada Realinho dkk. [2] dan Martins dkk. [3]. Seperti [3], satu
skenario hanya memakai informasi saat pendaftaran; seperti [2], skenario lain menambahkan
kinerja akademik semester 1 dan 2. Ketidakseimbangan kelas ditangani dengan pembobotan kelas
dan F1 dipakai sebagai metrik utama. Berbeda dari keduanya, target dibuat biner
(dropout vs bukan dropout) karena tujuan praktisnya menandai mahasiswa berisiko.

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
[1] V. Realinho, M. V. Martins, J. Machado, and L. Baptista, "Predict Students' Dropout and
Academic Success," UCI Machine Learning Repository, 2021. doi: 10.24432/C5MC89.

[2] V. Realinho, J. Machado, L. Baptista, and M. V. Martins, "Predicting Student Dropout and
Academic Success," *Data*, vol. 7, no. 11, p. 146, 2022. doi: 10.3390/data7110146.

[3] M. V. Martins, D. Tolledo, J. Machado, L. M. T. Baptista, and V. Realinho, "Early
prediction of student's performance in higher education: a case study," in *Trends and
Applications in Information Systems and Technologies*, Springer, 2021.
doi: 10.1007/978-3-030-72657-7_16.