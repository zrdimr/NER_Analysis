import pandas as pd
import numpy as np
import random

# Base Indonesian phrases for stress/non-stress
stress_phrases = [
    "Rasanya pengen nangis aja.",
    "Aku butuh bantuan, beban menyerang lagi.",
    "Sudah seminggu aku merasa putus asa.",
    "Pekerjaan bikin aku anxiety.",
    "Gak kuat lagi sama tekanan ini.",
    "Aku merasa sangat kacau akhir-akhir ini.",
    "Aku merasa sangat hampa akhir-akhir ini.",
    "Rasanya pengen pergi aja untuk selamanya.",
    "Kenapa hidup ini begitu kacau?",
    "Orang-orang gak ngerti gimana rasanya sedih dan hancur.",
    "Tiap malam aku ga bisa tidur mikirin beban hidup.",
    "Kepalaku mau pecah rasanya dengan semua masalah ini.",
    "Aku lelah berpura-pura baik-baik saja.",
    "Semua terasa sia-sia, aku ingin nyerah.",
    "Kecemasanku kambuh lagi hari ini.",
    "Terlalu banyak pikiran merusak hariku.",
    "Setiap hari rasanya berat sekali untuk dijalani.",
    "Kenapa tidak ada yang peduli padaku?",
    "Aku benci diriku sendiri.",
    "Masa depanku terlihat sangat gelap dan suram."
]

non_stress_phrases = [
    "Lagi belajar kodingan, lumayan susah tapi asik.",
    "Akhirnya bisa jalan-jalan dengan tenang.",
    "Besok rencana mau belanja sama teman.",
    "Liburan ke gunung yuk.",
    "Besok rencana mau masak sama teman.",
    "Hari ini cuaca sangat bagus dan cerah.",
    "Jangan lupa makan ya gais!",
    "Liburan ke pantai sangat menyenangkan.",
    "Jangan lupa belanja ya gais!",
    "Baru saja makan bakso, enak banget!",
    "Semangat pagi! Semoga harimu indah.",
    "Wah filmnya ternyata sangat seru untuk ditonton.",
    "Pekerjaanku selesai lebih awal hari ini.",
    "Senangnya bisa berkumpul bersama keluarga.",
    "Akhirnya akhir pekan tiba, waktunya bersantai.",
    "Buku yang kubaca hari ini sangat menginspirasi.",
    "Besok aku akan lari pagi agar lebih sehat.",
    "Cuaca hari ini sangat mendukung untuk nongkrong.",
    "Aku merasa sangat produktif dan bahagia hari ini.",
    "Terima kasih atas bantuan teman-teman semua."
]

random.seed(42)
np.random.seed(42)

new_texts = []
new_labels = []

# Generate 5000 records
# We want imbalance: ~70% Non-Stress (0) and ~30% Stress (1)
for i in range(5000):
    k = np.random.randint(4, 8)
    # Determine the true label for this instance
    is_stress = np.random.rand() < 0.3
    
    if is_stress:
        # Majority stress phrases
        num_stress = np.random.randint(k//2 + 1, k + 1)
        num_non_stress = k - num_stress
    else:
        # Majority non-stress phrases
        num_non_stress = np.random.randint(k//2 + 1, k + 1)
        num_stress = k - num_non_stress
        
    phrases = []
    phrases.extend(np.random.choice(stress_phrases, size=num_stress, replace=True))
    phrases.extend(np.random.choice(non_stress_phrases, size=num_non_stress, replace=True))
    random.shuffle(phrases)
    
    merged_text = " ".join(phrases)
    label = 1 if is_stress else 0
    
    # Add small noise 5%
    if random.random() < 0.05:
        label = 1 - label
        
    new_texts.append(merged_text)
    new_labels.append(label)

df_new = pd.DataFrame({
    "text": new_texts,
    "label": new_labels
})

df_new.to_csv("data/raw/Vibree_indonesian_stress_dataset.csv", index=False)
print("Saved modified Vibree dataset (Indonesian) with", len(df_new), "rows.")
print("Distribution:")
print(df_new["label"].value_counts())
