# Tugas Besar 1 Machine Learning

## Deskripsi

Tugas Besar 1 IF3270 Pembelajaran Mesin ini bertujuan untuk mengimplementasikan Feed Forward Neural Network (FFNN) dalam menyelesaikan permasalahan klasifikasi. Proyek ini menggunakan dataset MNIST, yang terdiri dari gambar angka tulisan tangan dari 0 hingga 9, untuk melatih dan menguji model. Feed Forward Neural Network (FFNN) adalah salah satu jenis artificial neural network yang sering digunakan dalam berbagai aplikasi pembelajaran mesin. Dalam proyek ini, kami mengimplementasikan konsep dasar FFNN, termasuk arsitektur, fungsi aktivasi, dan backpropagation.

## Cara Setup dan Run Program

1. Buat virtual environment python

    ```bash
    python -m venv .venv
    ```

2. Aktifkan virtual environment

    ```bash
    .\.venv\Scripts\activate
    ```

3. Install library yang dibutuhkan

    ```bash
    pip install -r requirements.txt
    ```

4. Install package local

    ```bash
    cd ./src/ffnn
    pip install -e .
    ```

5. Program sudah siap dijalankan dan dapat dilakukan development

## Anggota Kelompok

| Nama                        | NIM      | Pembagian Tugas                                                                                                                                        |
|-----------------------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ahmad Naufal Ramadan        | 13522005 | Inisialisasi struktur proyek, implementasi fungsi aktivasi dan fungsi loss beserta turunannya, membantu implementasi forward dan backward, implementasi L1 dan L2 regularization              |
| Yusuf Ardian Sandi          | 13522015 | Implementasi visualisasi neural network dengan graph, membantu implementasi forward pada FFNN class, implementasi grafik penurunan loss terhadap epoch |
| Rayendra Althaf Taraka Noor | 13522107 | Implementasi visualisasi distribusi weights dan loss, membantu  implementasi backward, penjelasan implementasi dalam laporan                           |
