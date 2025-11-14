# Bearing Fault Detection + SHAP

Ringkasan singkat proyek explainable AI untuk deteksi kerusakan bearing dengan dataset CWRU. Notebook utama `crwu-bearings-benchmarks-shap-explainability.ipynb` menunjukkan bagaimana kombinasi **SVC** (model non-linear) dan **Regresi Logistik** (model linear) dapat dijelaskan secara transparan memakai SHAP.

## Highlights

- **Dataset**: 2.300 sampel, 10 kelas (Normal, Ball, Inner Race, Outer Race) dengan 9 fitur statistik time-domain (max, min, mean, sd, rms, skewness, kurtosis, crest, form).
- **Model**:  
  - SVC (RBF + GridSearchCV) → **96,4%** akurasi test (723/750 benar).  
  - Logistic Regression (multinomial) → **94,3%** akurasi test.  
  - Subset 200 sampel untuk SHAP mencapai **98%** akurasi sehingga cukup representatif.
- **SHAP**: KernelExplainer untuk SVC dan LinearExplainer untuk LR. Visual utama: summary plot, stacked bar feature importance per-class, waterfall lokal, analisis kesalahan.
- **Insight inti**:
  1. **Mean & Kurtosis** = indikator universal; SVC juga mengandalkan crest/min, LR fokus pada mean/sd/rms.
  2. **Inner Race** → monitor mean; **Outer Race** → amati kurtosis/crest; **Ball Fault** → kombinasi mean+kurtosis.
  3. Kesalahan sisa hanya terjadi pada keparahan mirip (0.014" vs 0.021"); tidak ada kebingungan lintas lokasi.
  4. Strategi deployment: gunakan LR untuk monitoring cepat, eskalasi ke SVC saat risiko tinggi; pertimbangkan ensemble.

## Cara Pakai

```bash
pip install -r requirements.txt
jupyter notebook crwu-bearings-benchmarks-shap-explainability.ipynb
```
Jalankan notebook dari atas ke bawah (estimasi 20–30 menit termasuk SHAP).

## Struktur Singkat

```
project/
├── README.md
├── crwu-bearings-benchmarks-shap-explainability.ipynb
└── kaggle/input/cwru-bearing-datasets/
    ├── feature_time_48k_2048_load_1.csv
    └── raw/*.mat
```

## Kontribusi

- Pipeline SHAP multi-class (Kernel + Linear Explainer).
- Perbandingan mendalam SVC vs LR + potensi ensemble.
- Analisis fault-specific & severity trend untuk kebutuhan predictive maintenance.

## Lisensi & Referensi

- Data: Case Western Reserve University Bearing Data Center.  
- SHAP: Lundberg & Lee (NIPS 2017).  
- ML Tools: scikit-learn.  
Gunakan data sesuai ketentuan sumber.

## 📝 License

This project uses the CWRU Bearing Dataset, which is publicly available for research purposes. Please refer to the dataset documentation for usage terms.

## 👥 Acknowledgments

- **Case Western Reserve University** for providing the bearing dataset
- **SHAP developers** for the excellent explainability framework
- **Scikit-learn** team for comprehensive ML tools

---

**Note**: This project demonstrates that machine learning for industrial diagnostics can be both highly accurate (96.4% on full test set) and fully interpretable through SHAP. The insights gained can directly improve bearing monitoring systems, reduce unplanned downtime, and enhance predictive maintenance strategies.

For detailed analysis and visualizations, please refer to the Jupyter notebook: `crwu-bearings-benchmarks-shap-explainability.ipynb`

