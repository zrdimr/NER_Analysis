import numpy as np
import random
import re

# Custom Entity/Keyphrase to Text Dictionary for Indonesian Stress Augmented Data
ENTDA_DICT = {
    # Stress entities
    "nangis": ["menangis", "berurai air mata", "bersedih", "meratap"],
    "stress": ["tertekan", "depresi", "kalut", "stres berat"],
    "anxiety": ["cemas", "gelisah", "gugup", "was-was"],
    "kacau": ["berantakan", "hancur", "bermasalah", "buruk"],
    "beban": ["tanggungan", "beban hidup", "penderitaan", "masalah berat"],
    "hampa": ["kosong", "sepi", "tiada arti", "hambar"],
    "putus asa": ["menyerah", "hilang harapan", "pasrah", "lelah"],
    "lelah": ["capek", "penat", "habis tenaga"],
    "benci": ["tidak suka", "muak", "marah pada"],
    "gelap": ["buram", "suram", "tanpa cahaya"],
    
    # Non-stress entities
    "teman": ["kawan", "sahabat", "rekan", "kolega"],
    "pekerjaan": ["tugas", "pekerjaan kantor", "proyek", "tanggung jawab"],
    "belanja": ["beli barang", "shopping", "ke mall"],
    "liburan": ["jalan-jalan", "wisata", "rekreasi"],
    "gunung": ["alam terbuka", "bukit", "hutan"],
    "pantai": ["laut", "pulau", "pesisir"],
    "makan": ["sarapan", "makan siang", "makan malam"],
    "enak": ["lezat", "nikmat", "mantap"],
    "seru": ["asik", "menyenangkan", "menarik"],
    "bahagia": ["senang", "gembira", "bersukacita"],
    "produktif": ["aktif", "rajin", "semangat"]
}

def augment_text_entda(text):
    """
    Replaces recognized entities/keywords with synonyms (EnTDA).
    """
    words = text.split()
    augmented_words = []
    changed = False
    
    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word.lower())
        
        if clean_word in ENTDA_DICT and random.random() < 0.6: # 60% chance to replace if match
            synonym = random.choice(ENTDA_DICT[clean_word])
            
            # Match capitalization
            if word.istitle():
                synonym = synonym.title()
            
            # Keep punctuation attached
            prefix = word[:word.lower().find(clean_word)] if clean_word in word.lower() else ""
            suffix = word[word.lower().find(clean_word) + len(clean_word):] if clean_word in word.lower() else ""
            
            augmented_words.append(prefix + synonym + suffix)
            changed = True
        else:
            augmented_words.append(word)
            
    # Fallback to random deletion/insertion if no entity matched to guarantee variance
    if not changed and len(words) > 3:
        if random.random() < 0.5:
            # Drop a random word
            idx = random.randint(0, len(words) - 1)
            augmented_words.pop(idx)
        else:
            # Swap two adjacent words
            idx = random.randint(0, len(augmented_words) - 2)
            augmented_words[idx], augmented_words[idx+1] = augmented_words[idx+1], augmented_words[idx]
            
    return " ".join(augmented_words)

def augment_with_entda(texts, labels, target_balance=True):
    """
    Entity-to-Text Based Data Augmentation (EnTDA).
    Checks class imbalance and augments the minority class to match the majority class.
    """
    # Count classes
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    if len(class_counts) < 2:
        return texts, labels
        
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)
    
    majority_count = class_counts[majority_class]
    minority_count = class_counts[minority_class]
    
    diff = majority_count - minority_count
    
    if diff <= 0:
        return texts, labels
        
    print(f"EnTDA: Augmenting class {minority_class} by {diff} samples...")
    
    # Get all texts of minority class
    minority_texts = [texts[i] for i in range(len(texts)) if labels[i] == minority_class]
    
    augmented_texts = list(texts)
    augmented_labels = list(labels)
    
    for i in range(diff):
        # Randomly choose a text from minority class to augment
        idx = random.randint(0, len(minority_texts) - 1)
        original_text = minority_texts[idx]
        
        aug_text = augment_text_entda(original_text)
                
        augmented_texts.append(aug_text)
        augmented_labels.append(minority_class)
            
        if (i+1) % 500 == 0:
            print(f"Generated {i+1}/{diff} augmented samples...")
            
    print(f"EnTDA Complete. Total samples now: {len(augmented_texts)}")
    return augmented_texts, augmented_labels
