<!-- 36b26ff7-6d3d-4ba2-9371-f47458185335 9d8a2476-af24-4539-a83e-ada604bcc5f7 -->
# Rencana Perbaikan Notebook Penelitian CWRU Bearing

## Ringkasan

Membersihkan dan mengoptimalkan notebook dengan menghapus redundansi, mengonsolidasikan visualisasi SHAP, dan menyederhanakan alur narasi dari 89 sel menjadi ~50-55 sel.

## Fase 1: Hapus Sel Redundan (Cells to Delete)

### A. EDA yang Berlebihan

- **Cell 9**: Hapus correlation matrix heatmap - tidak dirujuk dalam findings
- **Cell 10**: Hapus feature distributions (9 features x 10 fault types) - terlalu banyak plot
- **Cell 11**: Hapus feature statistics by fault type - redundan dengan cell 8

### B. Confusion Matrix Duplikat  

- **Cell 24-25**: Hapus confusion matrix untuk baseline (unoptimized) model
- Pertahankan hanya cell 52-53 (final tuned model confusion matrix)

### C. Visualisasi Logistic Regression

- **Cell 79**: Hapus confusion matrix terpisah untuk LR
- Akan digabung dengan perbandingan model

## Fase 2: Konsolidasikan SHAP Analysis (Reduce from ~40 cells to 10-12)

### A. SHAP untuk SVC (Keep 3 visualizations)

- **Cell 43**: Pertahankan 1 SHAP summary plot (beeswarm) untuk SVC
- **Cell 44**: Hapus SHAP bar plot (redundan dengan cell 43)
- **Cell 47**: Kurangi waterfall plots dari banyak contoh menjadi 2 contoh saja
- **Cell 48**: Hapus waterfall untuk misclassified samples
- **Cell 49**: Hapus partial dependence plots (terlalu detail)

### B. SHAP untuk Logistic Regression (Keep 1 visualization)

- **Cell 60**: Pertahankan 1 summary plot untuk LR
- **Cell 61**: Hapus bar plot untuk LR (redundan)
- **Cell 57**: Hapus atau gabungkan dengan cell inisialisasi SHAP

### C. Perbandingan Model

- **Cell 63**: Pertahankan dan perbaiki comparison plot
- **Cell 64**: Hapus scatter plot korelasi (kurang informatif)
- **Cell 65**: Hapus identifikasi common features (pindahkan ke teks)

### D. Fault-Specific Analysis

- **Cell 67**: Konsolidasikan fault importance analysis
- **Cell 69**: Hapus atau gabungkan visualisasi berulang
- **Cell 70**: Hapus severity comparison plots (redundan)
- **Cell 71**: Hapus SHAP summary per fault type (terlalu banyak)

### E. Misclassification Analysis

- **Cell 73-74**: Hapus atau gabungkan analisis misclassification yang repetitif
- **Cell 76**: Hapus detailed misclassified sample analysis

## Fase 3: Konsolidasikan ROC/PR Curves

### A. Gabungkan dalam Satu Plot

- **Cell 78**: Modifikasi untuk menggabungkan ROC curves untuk SVC dan LR dalam satu plot
- **Cell 79**: Modifikasi untuk menggabungkan PR curves untuk SVC dan LR dalam satu plot
- Target: 2 plots total (1 ROC, 1 PR) instead of 4 separate plots

## Fase 4: Perbaiki Struktur Markdown

### A. Ringkasan Eksekutif (Cell 1)

- Pertahankan apa adanya - sudah bagus

### B. Simplifikasi Section Headers

- **Cell 42**: Hapus "Analisis Subset SHAP (200 sampel)" - gabungkan ke header utama
- **Cell 68**: Hapus "Full Test Set Performance Analysis" - redundan
- **Cell 75**: Hapus "Perbandingan Performa: Set Test Lengkap vs Subset SHAP"

### C. Findings Section (Cells 81-88)

- **Cell 82-84**: Konsolidasikan wawasan ke dalam struktur yang lebih ringkas
- **Cell 85**: Hapus "Detail Analisis Tambahan" - redundan
- **Cell 86**: Hapus "Wawasan Visualisasi" - redundan

## Fase 5: Update Metadata & Numbering

- Update cell execution_count untuk konsistensi
- Pastikan tidak ada broken references antar sel
- Verify bahwa semua variabel yang dibutuhkan masih terdefined

## Target Akhir

**Sebelum**: 89 cells (54 code, 35 markdown)
**Sesudah**: ~50-55 cells (~30 code, ~20-25 markdown)

**Struktur Final**:

1. Ringkasan Eksekutif
2. Identifikasi Jenis Kerusakan  
3. Mengambil Data
4. EDA Sederhana (tanpa correlation matrix & excessive plots)
5. Train/Test Split & Scaling
6. Model Training & Tuning
7. Evaluasi Model Final (1 confusion matrix, 1 ROC, 1 PR)
8. SHAP Analysis (3-4 key plots only)
9. Temuan Kunci (consolidated)
10. Comprehensive Classification Report

**Pengurangan**:

- 15-20 sel visualisasi redundan
- 10-12 sel SHAP yang berlebihan
- 5-6 sel EDA yang terlalu detail