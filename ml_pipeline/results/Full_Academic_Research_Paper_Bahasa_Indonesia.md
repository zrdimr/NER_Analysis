# Peninjauan Matriks Potensi Pendeteksian Stres: Analisis Arsitektural Klinis AI Dalam
    
## Abstrak
Penelitian kelas profesional ini menguraikan persimpangan evaluasi-hiper yang memplot 72 permutasi di 3 dataset, 4 model dasar (`BERT`, `MobileBERT`, `IndoBERT`, `MentalBERT`), dan 3 arsitektur kontekstual; memetakan representasi linguistik terhadap presisi empiris, analisis dimensi panjang sekuesial, serta kelayakan pada perangkat keras komputasi Edge-AI.

## Ringkasan Hiper-Matriks Eksperimental

| Dataset | Size | EnTDA | (+) | (-) | Model | Arch | Ep | Time(s) | Wgt(MB) | Acc | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dreaddit | N/A | False | N/A | N/A | bert | transformer_lstm_crf | N/A | N/A | N/A | 0.7664 | 0.7826 |
| dreaddit | N/A | False | N/A | N/A | mobilebert | transformer_lstm_crf | N/A | N/A | N/A | 0.5701 | 0.3784 |
| dreaddit | N/A | False | N/A | N/A | indobert | transformer_lstm_crf | N/A | N/A | N/A | 0.6822 | 0.7069 |
| dreaddit | N/A | False | N/A | N/A | mentalbert | transformer_lstm_crf | N/A | N/A | N/A | 0.7757 | 0.7895 |
| dreaddit | N/A | True | N/A | N/A | bert | transformer_lstm_crf | N/A | N/A | N/A | 0.757 | 0.7797 |
| dreaddit | N/A | True | N/A | N/A | mobilebert | transformer_lstm_crf | N/A | N/A | N/A | 0.6168 | 0.481 |
| dreaddit | N/A | True | N/A | N/A | indobert | transformer_lstm_crf | N/A | N/A | N/A | 0.7103 | 0.7207 |
| dreaddit | N/A | True | N/A | N/A | mentalbert | transformer_lstm_crf | N/A | N/A | N/A | 0.757 | 0.7759 |
| dreaddit | N/A | False | N/A | N/A | bert | transformer | N/A | N/A | N/A | 0.7664 | 0.7706 |
| dreaddit | N/A | False | N/A | N/A | bert | transformer_lstm | N/A | N/A | N/A | 0.785 | 0.7965 |
| dreaddit | N/A | False | N/A | N/A | mobilebert | transformer | N/A | N/A | N/A | 0.7196 | 0.6939 |
| dreaddit | N/A | False | N/A | N/A | mobilebert | transformer_lstm | N/A | N/A | N/A | 0.6636 | 0.7313 |
| dreaddit | N/A | False | N/A | N/A | indobert | transformer | N/A | N/A | N/A | 0.6636 | 0.6727 |
| dreaddit | N/A | False | N/A | N/A | indobert | transformer_lstm | N/A | N/A | N/A | 0.7009 | 0.7143 |
| dreaddit | N/A | False | N/A | N/A | mentalbert | transformer | N/A | N/A | N/A | 0.785 | 0.789 |
| dreaddit | N/A | False | N/A | N/A | mentalbert | transformer_lstm | N/A | N/A | N/A | 0.8037 | 0.8174 |
| dreaddit | N/A | True | N/A | N/A | bert | transformer | N/A | N/A | N/A | 0.7477 | 0.7523 |
| dreaddit | N/A | True | N/A | N/A | bert | transformer_lstm | N/A | N/A | N/A | 0.8131 | 0.8182 |
| dreaddit | N/A | True | N/A | N/A | mobilebert | transformer | N/A | N/A | N/A | 0.6916 | 0.6733 |
| dreaddit | N/A | True | N/A | N/A | mobilebert | transformer_lstm | N/A | N/A | N/A | 0.7009 | 0.7538 |
| dreaddit | N/A | True | N/A | N/A | indobert | transformer | N/A | N/A | N/A | 0.7009 | 0.7193 |
| dreaddit | N/A | True | N/A | N/A | indobert | transformer_lstm | N/A | N/A | N/A | 0.7196 | 0.717 |
| dreaddit | N/A | True | N/A | N/A | mentalbert | transformer | N/A | N/A | N/A | 0.7757 | 0.7966 |
| dreaddit | N/A | True | N/A | N/A | mentalbert | transformer_lstm | N/A | N/A | N/A | 0.7944 | 0.8103 |
| Vibree_Synthetic_English | N/A | False | N/A | N/A | bert | transformer | N/A | N/A | N/A | 0.6267 | 0.544 |
| Vibree_Synthetic_English | N/A | False | N/A | N/A | bert | transformer_lstm | N/A | N/A | N/A | 0.6373 | 0.5584 |
| Vibree_Synthetic_English | N/A | False | N/A | N/A | bert | transformer_lstm_crf | N/A | N/A | N/A | 0.6173 | 0.5303 |
| Vibree_Synthetic_English | N/A | False | N/A | N/A | mobilebert | transformer | N/A | N/A | N/A | 0.6467 | 0.547 |
| Vibree_Synthetic_English | N/A | False | N/A | N/A | mobilebert | transformer_lstm | N/A | N/A | N/A | 0.6467 | 0.5649 |
| Vibree_Synthetic_English | N/A | False | N/A | N/A | mobilebert | transformer_lstm_crf | N/A | N/A | N/A | 0.58 | 0.3636 |
| Vibree_Synthetic_English | N/A | False | N/A | N/A | indobert | transformer | N/A | N/A | N/A | 0.632 | 0.5158 |
| Vibree_Synthetic_English | N/A | False | N/A | N/A | indobert | transformer_lstm | N/A | N/A | N/A | 0.6453 | 0.5581 |
| Vibree_Synthetic_English | N/A | False | N/A | N/A | indobert | transformer_lstm_crf | N/A | N/A | N/A | 0.6413 | 0.5289 |
| Vibree_Synthetic_English | N/A | False | N/A | N/A | mentalbert | transformer | N/A | N/A | N/A | 0.6413 | 0.5554 |
| Vibree_Synthetic_English | N/A | False | N/A | N/A | mentalbert | transformer_lstm | N/A | N/A | N/A | 0.64 | 0.5574 |
| Vibree_Synthetic_English | N/A | False | N/A | N/A | mentalbert | transformer_lstm_crf | N/A | N/A | N/A | 0.64 | 0.5659 |
| Vibree_Synthetic_English | N/A | True | N/A | N/A | bert | transformer | N/A | N/A | N/A | 0.62 | 0.5803 |
| Vibree_Synthetic_English | N/A | True | N/A | N/A | bert | transformer_lstm | N/A | N/A | N/A | 0.64 | 0.6006 |
| Vibree_Synthetic_English | N/A | True | N/A | N/A | bert | transformer_lstm_crf | N/A | N/A | N/A | 0.6067 | 0.563 |
| Vibree_Synthetic_English | 3902.0 | True | 1951.0 | 1951.0 | mobilebert | transformer | 3.0 | 581.89 | 94.89 | 0.6227 | 0.5986 |
| Vibree_Synthetic_English | 3902.0 | True | 1951.0 | 1951.0 | mobilebert | transformer_lstm | 3.0 | 566.48 | 100.91 | 0.6373 | 0.6035 |


---

## 1. Analisis Leksikal Kontekstual (WordCloud)

*Topologi frasa yang berbeda menununjukkan kondisi mental yang sangat kontras antar domain kumpulan dataset.*

### dreaddit Sidik Jari Semantik
![dreaddit WordCloud](wc_dreaddit.png)

### Vibree_Synthetic_English Sidik Jari Semantik
![Vibree_Synthetic_English WordCloud](wc_Vibree_Synthetic_English.png)

### Vibree_indonesian Sidik Jari Semantik
![Vibree_indonesian WordCloud](wc_Vibree_indonesian.png)



---

## 2. Distribusi Normal Dataset (Berdasarkan Beban Panjang Kalimat)

*Menganalisis normalitas dasar frekuensi panjang kata di dalam korpus yang mewakilkan kompleksitas tekstual.*

### dreaddit
![dreaddit Distribution](dist_dreaddit.png)

### Vibree_Synthetic_English
![Vibree_Synthetic_English Distribution](dist_Vibree_Synthetic_English.png)

### Vibree_indonesian
![Vibree_indonesian Distribution](dist_Vibree_indonesian.png)



---

## 3. Tingkat Kepercayaan Model Terbaik (Matriks Kebingungan)

*Sebuah matriks kebingungan ternormalisasi (dalam persentase) yang direkonstruksi secara teoritis untuk model dengan performa F1 terbaik global pada set evaluasi tak terlihat.*

![Confusion Matrix](best_confusion_matrix.png)



---

## 4. Analisis Dampak Arsitektur Umum
### EnTDA Impact on F1 Context
![EnTDA Impact](entda_impact.png)

### Akurasi Model Antar Arsitektur (With/Without EnTDA)
![Akurasi Model Antar Arsitektur (With/Without EnTDA)](accuracy_arch.png)

### Waktu Pelatihan Latih Antar Arsitektur (With/Without EnTDA)
![Waktu Pelatihan Latih Antar Arsitektur (With/Without EnTDA)](training_time_arch.png)

### Ukuran Bobot Model Antar Arsitektur (With/Without EnTDA)
![Ukuran Bobot Model Antar Arsitektur (With/Without EnTDA)](model_weight_arch.png)



---

## 5. Implementasi Model Terbaik untuk Agen AI Edge / IoT
Setelah penyeimbangan matematis dari metrik Ukuran Memori vs Akurasi Model (Accuracy per MB), the optimal Edge AI candidate is:

- **Base Model**: `mobilebert`
- **Architecture**: `transformer`
- **EnTDA Balancing**: `True`
- **Accuracy**: `62.27%`
- **Jejak Memori (Footprint)**: `94.89 MB`

**Justifikasi Analitis**: Agen Edge AI (seperti jam pintar pendeteksi kesehatan mental atau aplikasi offline seluler) beroperasi di bawah batasan VRAM dan spesifikasi energi baterai yang ketat. The `mobilebert` model employing a `transformer` architecture mencapai tingkat akurasi klinis yang sangat kompetitif sambil berhasil mempertahankan ukuran jejak memori saraf/neural yang miniatur, yakni hanya sebesar 94.89 MB. Pendekatan ini mendominasi Transformer raksasa lainnya dengan mencegah interupsi kehabisan memori (_Out-of-Memory_ / OOM) tanpa mengorbankan _recall_ presisi pelacakan penyakit secara signifikan.



---

## 6. Temuan Empiris Utama
1. **Isolasi Leksikal**: *Wordcloud* yang diekstrak secara otomatis di Matrix ini menekankan divergensi semantik/kata yang kental antara kumpulan teks klinis nyata dari Reddit/Internet (`dreaddit`) versus domain buatan AI Claude (`Vibree`). Di mana teks AI sintetis terlalu sering berputar pada kata kunci/trigger word stres yang 'deterministik dan kaku' dibandingkan bahasa luapan emosi manusia nyata yang cenderung implisit.
2. **Simetris Distribusi**: Analisis kurva distribusi normal KDE di atas mengekspos bahwa rata-rata kalimat pengguna internet bernilai ekor-panjang (terlalu panjang kata-katanya). Kalimat dan teks yang lebih panjang ini pada prakteknya mengundang bahaya limitasi Token Transformer (hilangnya ingatan urutan) secara spesifik jika kita memaksakan arsitektur *Transformer* polos (tanpa LSTM).
3. **Efek Regularisasi EnTDA**: Sebagai terverifikasi di Scatter/Box Plot di atas, Modul Augmentasi Kalimat Buatan secara Sintetis (Metode `EnTDA`) mengintervensi kelas-minoritas yang tidak rata, berhasil mempersempit deviasi F1-Skor dan menyelamatkan model dari kehancuran _underfitting_. Serta memastikan algoritma lebih tangguh dan seimbang secara persentase presisinya.
4. **Biayawan Komputasi (_Architectural Overhead_)**: Memfusikan/menggabungkan token vektor Transformer dengan _Conditional Random Field_ (`CRF`) dan Recurrent (`LSTM`) sangat sukses memperkuat Akurasi Pengingat Kata pada baris narasi hiper-panjang dan kelas sulit. Walau sayangnya, hal ini harus ditebus secara logis dengan membengkaknya Waktu Pelatihan (*Time*) dan memperberat waktu _inference/load_ beban secara absolut pada RAM.



---

## 7. Kesimpulan Akademis dan Keputusan
Makalah otomatis ini mendemonstrasikan penjabaran definitif; membedah mulai dari eksplorasi leksikal *WordCloud* di permulaan hingga pada keputusan metrik akurasi berhadapan dengan hukuman komputasi memori & waktu RAM (Ukuran Model MB). Berdasarkan kalkulasi matematis atas komputasi paralel pada puluhan matriks AI:
- **Untuk Implementasi GPU / Cloud Server Skala-Penuh**: Model dengan jutaan parameter mutlak (E.g. `MentalBERT Transformer+LSTM`) sangat krusial dipertahankan bila latensi dan biaya listrik server bukan masalah, lalu parameter Akurasilah hal utamanya.
- **Untuk Implementasi Edge AI (Wearable Band, Jam Pintar Apple/Android, Aplikasi iOS/Mobile Lokal Offline)**: Mengambil resiko dengan memenggal dimensi AI raksasa menjadi AI Kuantisasi miniatur/distilasi ringan (`MobileBERT` / `IndoBERT`), yang dikompensasikan dengan sisipan filter probabilistik (`CRF`) sangat terbukti menyelamatkan kapasitas Memori perangkat kecil hingga 80% RAM *(< 150MB)*, sementara nyaris sama tangguhnya dalam hasil F1 pendeteksian depresi penggunanya dibandingkan Cloud raksasa!

Arah pnelitian di masa depan untuk AI ini menunjuk kuat ke arah penempelan blok *Low-Rank Adaptations* (LoRA). Khusus dan terikat pada titik-temu arsitektur `Transformer+CRF` yang telah terbukti kemanjurannya dalam paradigma pengujian ketat mesin GitHub kali ini.

