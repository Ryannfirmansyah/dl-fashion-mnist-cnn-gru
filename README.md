# Tugas Deep Learning: CNN vs BiGRU pada Fashion-MNIST

Tugas individu mata kuliah Deep Learning, Universitas Hasanuddin.

- **Nama:** Ryan Firmansyah
- **NIM:** H071241082
- **Prodi:** Sistem Informasi, Departemen Matematika, FMIPA, Universitas Hasanuddin

## Ringkasan
Dua arsitektur dilatih pada dataset yang sama lalu dibandingkan:

1. **CNN**: 4 lapis Conv2D (32, 32, 64, 64 filter) + BatchNorm + MaxPool + Dropout, Dense 128, softmax 10 kelas.
2. **BiGRU (RNN)**: gambar dibaca sebagai sequence 28 baris × 28 fitur. Bidirectional GRU (96) → GRU (64) → Dense, softmax 10 kelas.

**Dataset:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), 70.000 citra grayscale 28×28 dalam 10 kelas pakaian (60.000 train, 10.000 test). Dari data train, 10% dipakai sebagai validasi (stratified).

## Hasil (data test, 10.000 citra)

| Model | Test accuracy | Macro F1 | Parameter | Waktu training (CPU 2 core) |
|---|---|---|---|---|
| CNN | **92,62%** | 0,9255 | 468.394 | 912 s |
| BiGRU | 89,00% | 0,8894 | 126.922 | 469 s |

Temuan utama: CNN lebih akurat karena memanfaatkan struktur spasial 2D; BiGRU jauh lebih ringan tetapi lebih rendah akurasinya. Kelas tersulit untuk kedua model adalah *Shirt* (sering tertukar dengan T-shirt/top, Coat, Pullover). Analisis lengkap ada di bagian akhir notebook.

Pengaturan: Adam (lr 1e-3), batch 128, 12 epoch, EarlyStopping + ReduceLROnPlateau, seed 42, satu kali run.

## Struktur Repo
```
.
├── tugas_deep_learning.ipynb   # notebook lengkap (sudah berisi output)
├── requirements.txt
├── results/
│   ├── metrics.json            # metrik kedua model
│   ├── training_curves.png
│   ├── confusion_matrices.png
│   ├── cnn_misclassified.png
│   ├── sample_images.png
│   ├── cnn_fashion_mnist.keras
│   └── bigru_fashion_mnist.keras
└── README.md
```

## Cara Menjalankan
```bash
pip install -r requirements.txt
jupyter notebook tugas_deep_learning.ipynb
```
Dataset diunduh otomatis dari GitHub resmi Fashion-MNIST ke folder `data/` saat sel pemuatan data dijalankan (butuh internet). Training di CPU memakan waktu sekitar 20 menit.

## Catatan
Transfer Learning tidak dikerjakan; ketentuan tugas hanya meminta minimal 2 arsitektur, sehingga CNN dan RNN (GRU) dipilih.
