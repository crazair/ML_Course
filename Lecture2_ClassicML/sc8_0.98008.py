import os
import sys
from math import log2
import time
import pickle
import hashlib

import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score
from multiprocessing import Pool, cpu_count

# 🔥 PyTorch imports
import torch
import torch.nn as nn
import os
import multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import psutil

# ========================================
# 🔥 ULTRA OPTIMIZATION FOR INTEL ULTRA 9 285K + 96GB RAM
# ========================================
# ===== CPU Threading =====
# Используем все 24 ядра максимально эффективно
torch.set_num_threads(24)
torch.set_num_interop_threads(6)  # Параллелизм между операциями

# ===== Intel MKL Optimization =====
# Math Kernel Library - оптимизации для Intel CPU
os.environ['MKL_NUM_THREADS'] = '24'
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
os.environ['KMP_BLOCKTIME'] = '1'  # Быстрый контекстный переключатель
os.environ['MKL_DYNAMIC'] = 'FALSE'  # Фиксированное количество потоков

# ===== Memory Optimization =====
os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'  # Агрессивная очистка памяти
os.environ['MALLOC_MMAP_THRESHOLD_'] = '100000'

# ===== PyTorch Memory =====
# С 96GB можем позволить больше кэша
torch.backends.cudnn.benchmark = False  # Для CPU не нужно
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======================================================
# Constants
# ======================================================

COMMON_WORDS = set([
    'www','web','mail','shop','store','market','marketplace',
    'online','site','page','home','portal','hub',
    'app','apps','api','cdn','cloud','server','host','hosting',
    'tech','it','ai','dev','labs','software','system','systems',
    'data','network','net','connect','link','platform','tools',
    'media','news','blog','info','search','seo','marketing',
    'pro','plus','group','company','business','global','world',
    'service','services','solutions','support','help','secure','security',
    'login','auth','sign','account','user','admin','my','get','go',
    'buy','pay','finance','bank','capital','trade','invest','money',
    'mobile','smart','digital','studio','creative','design',
    'retail','auto','car','travel','booking',
    'food','health','care','life','free','best','top','new',
    'game','games','play','video','music','photo','image','file',
    'download','upload','share','social','chat','forum','community',
    'register','member','premium','vip','gold','silver','elite',
    'learn','education','course','training','tutorial','guide',
    'real','estate','property','rent','sale','deal','offer',
    'fashion','style','beauty','luxury','brand','official',
    'local','city','state','country','region','area','zone',
    'express','fast','quick','instant','direct','easy','simple',
    'happy','fun','cool','awesome','great','super','mega','ultra'
])

VOWELS = set("aeiou")

COMMON_TLDS = set([
    'com','org','net','edu','gov','io','co','ai','app','dev',
    'tech','online','store','shop','site','web','cloud','info',
    'biz','us','uk','de','jp','cn','ru','br','in','au','ca','fr','es'
])

COMMON_BIGRAMS = set([
    'th','he','in','er','an','re','on','at','en','nd','ti','es','or','te','of',
    'ed','is','it','al','ar','st','to','nt','ng','se','ha','as','ou','io','le'
])

COMMON_TRIGRAMS = set([
    'the','and','ing','ion','tio','ent','for','her','ter','hat','tha','ere',
    'ate','his','con','res','ver','all','ons','nce','men','ith','ted','ers'
])

# 🔥 Neural network parameters
MAX_LEN = 64
CHAR_VOCAB_SIZE = 40


# ======================================================
# Feature Engineering Functions
# ======================================================

def contains_common_word(name):
    return int(any(w in name for w in COMMON_WORDS))

def shannon_entropy(s):
    if not s:
        return 0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * log2(p) for p in probs)

def digit_ratio(name):
    return sum(c.isdigit() for c in name) / max(1, len(name))

def vowel_ratio(name):
    letters = sum(c.isalpha() for c in name)
    vowels = sum(c in VOWELS for c in name)
    return vowels / max(1, letters)

def clean_entropy(name):
    clean = ''.join(c for c in name if c.isalpha())
    return shannon_entropy(clean)

def max_consonant_run(name):
    run = max_run = 0
    for c in name:
        if c.isalpha() and c not in VOWELS:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run

def char_diversity(name):
    return len(set(name)) / max(1, len(name))

def bigram_diversity(name):
    if len(name) < 2:
        return 0
    b = [name[i:i+2] for i in range(len(name)-1)]
    return len(set(b)) / len(b)

def type_switch_ratio(name):
    def t(c):
        if c.isdigit():
            return "d"
        if c.isalpha():
            return "a"
        return "o"
    switches = sum(t(name[i]) != t(name[i-1]) for i in range(1, len(name)))
    return switches / max(1, len(name)-1)

def has_valid_tld(domain):
    parts = domain.split('.')
    if len(parts) < 2:
        return 0
    return int(parts[-1].lower() in COMMON_TLDS)

def repeated_pattern_score(name):
    if len(name) < 4:
        return 0
    max_repeat = 0
    for pattern_len in range(2, min(6, len(name)//2 + 1)):
        for i in range(len(name) - pattern_len * 2 + 1):
            pattern = name[i:i+pattern_len]
            count = 1
            j = i + pattern_len
            while j + pattern_len <= len(name) and name[j:j+pattern_len] == pattern:
                count += 1
                j += pattern_len
            max_repeat = max(max_repeat, count)
    return max_repeat

def pronounceability_score(name):
    if not name:
        return 0
    valid_patterns = 0
    for i in range(len(name) - 1):
        c1, c2 = name[i], name[i+1]
        if c1.isalpha() and c2.isalpha():
            if (c1 in VOWELS and c2 not in VOWELS) or (c1 not in VOWELS and c2 in VOWELS):
                valid_patterns += 1
    return valid_patterns / max(1, len(name) - 1)

def rare_char_ratio(name):
    rare_chars = set('qxzjkv')
    return sum(c in rare_chars for c in name.lower()) / max(1, len(name))

def trigram_diversity(name):
    if len(name) < 3:
        return 0
    tg = [name[i:i+3] for i in range(len(name)-2)]
    return len(set(tg)) / len(tg) if tg else 0

def common_bigram_ratio(name):
    if len(name) < 2:
        return 0
    bigrams = [name[i:i+2].lower() for i in range(len(name)-1) if name[i:i+2].isalpha()]
    if not bigrams:
        return 0
    return sum(bg in COMMON_BIGRAMS for bg in bigrams) / len(bigrams)

def common_trigram_ratio(name):
    if len(name) < 3:
        return 0
    trigrams = [name[i:i+3].lower() for i in range(len(name)-2) if name[i:i+3].isalpha()]
    if not trigrams:
        return 0
    return sum(tg in COMMON_TRIGRAMS for tg in trigrams) / len(trigrams)

def consonant_vowel_alternation(name):
    if len(name) < 2:
        return 0
    alternations = 0
    for i in range(len(name) - 1):
        c1, c2 = name[i].lower(), name[i+1].lower()
        if c1.isalpha() and c2.isalpha():
            if (c1 in VOWELS) != (c2 in VOWELS):
                alternations += 1
    return alternations / max(1, len(name) - 1)

def length_category(name):
    l = len(name)
    if l <= 5:
        return 1
    elif l <= 10:
        return 2
    elif l <= 15:
        return 3
    elif l <= 20:
        return 4
    else:
        return 5

def has_numbers_and_letters(name):
    has_digit = any(c.isdigit() for c in name)
    has_alpha = any(c.isalpha() for c in name)
    return int(has_digit and has_alpha)

def legit_score_composite(domain):
    name = domain.split('.')[0].lower()
    score = 0

    if contains_common_word(name):
        score += 3
    if has_valid_tld(domain):
        score += 2
    if len(name) <= 12:
        score += 1
    if vowel_ratio(name) > 0.3 and vowel_ratio(name) < 0.6:
        score += 1
    if max_consonant_run(name) <= 4:
        score += 1
    if pronounceability_score(name) > 0.4:
        score += 2
    if common_bigram_ratio(name) > 0.3:
        score += 2
    if common_trigram_ratio(name) > 0.2:
        score += 2

    if shannon_entropy(name) > 3.8:
        score -= 2
    if max_consonant_run(name) > 6:
        score -= 2
    if repeated_pattern_score(name) > 1:
        score -= 2
    if rare_char_ratio(name) > 0.15:
        score -= 1

    return score

def extract_features(domain):
    # 🔥 Защита от NaN и некорректных данных
    if not isinstance(domain, str) or not domain or pd.isna(domain):
        # Возвращаем нулевые признаки для некорректных доменов
        return [0] * 28  # 28 признаков в extract_features

    name = domain.split('.')[0].lower()
    return [
        len(name),
        sum(c.isdigit() for c in name),
        digit_ratio(name),
        sum(c.isalpha() for c in name),
        vowel_ratio(name),
        shannon_entropy(name),
        clean_entropy(name),
        max_consonant_run(name),
        char_diversity(name),
        bigram_diversity(name),
        type_switch_ratio(name),
        contains_common_word(name),
        has_valid_tld(domain),
        repeated_pattern_score(name),
        pronounceability_score(name),
        rare_char_ratio(name),
        trigram_diversity(name),
        int(len(name) > 15),
        int('.' in domain),
        common_bigram_ratio(name),
        common_trigram_ratio(name),
        legit_score_composite(domain),
        int(legit_score_composite(domain) >= 5),
        int(legit_score_composite(domain) <= -3),
        consonant_vowel_alternation(name),
        length_category(name),
        has_numbers_and_letters(name),
        int(len(name) <= 8 and legit_score_composite(domain) >= 3),
    ]

def extract_fp_features(domain):
    # 🔥 Защита от NaN и некорректных данных
    if not isinstance(domain, str) or not domain or pd.isna(domain):
        return [0] * 15  # 15 признаков в extract_fp_features

    name = domain.split('.')[0].lower()
    return [
        contains_common_word(name),
        int(len(name) <= 6),
        vowel_ratio(name),
        digit_ratio(name),
        shannon_entropy(name),
        max_consonant_run(name),
        has_valid_tld(domain),
        pronounceability_score(name),
        int(len(name) <= 10 and vowel_ratio(name) > 0.3),
        common_bigram_ratio(name),
        common_trigram_ratio(name),
        int(legit_score_composite(domain) >= 4),
        int(legit_score_composite(domain) >= 6),
        consonant_vowel_alternation(name),
        int(pronounceability_score(name) > 0.5),
    ]

def get_sld(d):
    # 🔥 Защита от NaN
    if not isinstance(d, str) or not d or pd.isna(d):
        return ""
    return str(d).split('.')[0].lower()


# ======================================================
# 🔥 PyTorch Neural Network Components
# ======================================================

def char_to_int(c):
    """Convert character to integer"""
    c = c.lower()
    if 'a' <= c <= 'z':
        return ord(c) - ord('a') + 1
    elif '0' <= c <= '9':
        return ord(c) - ord('0') + 27
    elif c == '.':
        return 37
    elif c == '-':
        return 38
    else:
        return 39

def encode_domain(domain):
    """Encode domain as sequence of integers"""
    encoded = [char_to_int(c) for c in domain[:MAX_LEN]]
    # Pad with zeros
    if len(encoded) < MAX_LEN:
        encoded += [0] * (MAX_LEN - len(encoded))
    return encoded

class DomainDataset(Dataset):
    """PyTorch Dataset for domains"""
    def __init__(self, domains, labels=None):
        self.domains = domains
        self.labels = labels

    def __len__(self):
        return len(self.domains)

    def __getitem__(self, idx):
        encoded = encode_domain(self.domains[idx])
        x = torch.tensor(encoded, dtype=torch.long)

        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
            return x, y
        return x

class CNNLSTMModel(nn.Module):
    """
    Hybrid CNN-LSTM model for DGA detection
    - CNN extracts local character patterns
    - BiLSTM captures sequential dependencies
    """
    def __init__(self, vocab_size=CHAR_VOCAB_SIZE, embedding_dim=64,
                 hidden_dim=64, dropout=0.4):
        super(CNNLSTMModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # CNN branch
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=4, padding=2)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

        # LSTM branch
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=dropout)

        # Combine branches
        self.fc1 = nn.Linear(64 + hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Embedding: (batch, seq_len) -> (batch, seq_len, emb_dim)
        emb = self.embedding(x)

        # CNN branch
        # Conv1d expects (batch, channels, seq_len)
        cnn_input = emb.permute(0, 2, 1)

        c1 = F.relu(self.bn1(self.conv1(cnn_input)))
        c2 = F.relu(self.bn2(self.conv2(c1)))
        c3 = F.relu(self.bn3(self.conv3(c2)))

        # Global max pooling
        cnn_out = F.max_pool1d(c3, c3.size(2)).squeeze(2)

        # LSTM branch
        lstm_out, _ = self.lstm(emb)
        # Take last output
        lstm_out = lstm_out[:, -1, :]

        # Concatenate
        combined = torch.cat([cnn_out, lstm_out], dim=1)

        # Dense layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))

        return x.squeeze()

def train_neural_model(train_domains, train_labels, val_domains, val_labels,
                      epochs=18, batch_size=512, lr=0.001):
    """
    Train PyTorch CNN-LSTM model
    OPTIMIZED FOR: Intel Ultra 9 285K + 96GB DDR5
    """
    print("\n" + "="*70)
    print("🔥 NEURAL NETWORK TRAINING (CPU OPTIMIZED)")
    print("="*70)

    # Create datasets
    train_dataset = DomainDataset(train_domains, train_labels)
    val_dataset = DomainDataset(val_domains, val_labels)

    # 🔥 ОПТИМИЗИРОВАННЫЕ DATALOADER'Ы
    # С 96GB RAM можем использовать много workers и prefetch
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,              # 🔥 16 процессов для загрузки данных
        pin_memory=False,            # Только для GPU
        persistent_workers=True,     # Переиспользуем workers
        prefetch_factor=4            # 🔥 Загружаем 4 батча заранее
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,   # 🔥 Валидация может больше
        shuffle=False,
        num_workers=12,
        persistent_workers=True,
        prefetch_factor=4
    )

    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Workers: 16 (train) + 12 (val)")
    print(f"✓ Prefetch factor: 4")
    print(f"✓ Total batches per epoch: {len(train_loader)}")

    # Create model
    model = CNNLSTMModel().to(device)

    # 🔥 INTEL OPTIMIZATION - Compile with TorchScript
    if device.type == 'cpu':
        try:
            model = torch.jit.script(model)
            print("✓ TorchScript compilation enabled (Intel MKL acceleration)")
        except Exception as e:
            print(f"⚠️  TorchScript compilation failed: {e}")
            print("   Continuing with regular model...")

    print(f"\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1e6:.2f} MB")

    # Loss and optimizer
    criterion = nn.BCELoss()

    # 🔥 AdamW с weight decay для лучшей генерализации
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # 🔥 Более агрессивный scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,      # Уменьшаем LR вдвое
        patience=2,      # Через 2 эпохи без улучшения
        min_lr=1e-6      # Минимальный LR
    )

    # Training loop
    print("\n" + "="*70)
    print("🔥 TRAINING START")
    print("="*70)

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5

    # 🔥 Мониторинг системы
    process = psutil.Process()

    for epoch in range(epochs):
        epoch_start = time.time()

        # ===== TRAINING =====
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (batch_x, batch_y) in enumerate(pbar):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)  # 🔥 Быстрее чем zero_grad()

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()

            # 🔥 Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Метрики
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)

            # Обновление прогресс-бара
            if batch_idx % 10 == 0:
                current_acc = 100 * train_correct / train_total
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.2f}%'
                })

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                predicted = (outputs > 0.5).float()
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        epoch_time = time.time() - epoch_start

        # 🔥 Системная статистика
        mem_info = process.memory_info()
        cpu_percent = process.cpu_percent()

        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train: Loss={avg_train_loss:.4f}, Acc={train_accuracy:.2f}%")
        print(f"  Val:   Loss={avg_val_loss:.4f}, Acc={val_accuracy:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  RAM: {mem_info.rss / 1e9:.2f} GB, CPU: {cpu_percent:.1f}%")

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if old_lr != new_lr:
            print(f"  📉 Learning rate: {old_lr:.6f} → {new_lr:.6f}")

        # Early stopping & model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Сохраняем лучшую модель
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
            }, 'best_nn_model.pt')

            print(f"  ✓ Best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ⚠️  No improvement ({patience_counter}/{max_patience})")

            if patience_counter >= max_patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break

    # Load best model
    print("\n" + "="*70)
    checkpoint = torch.load('best_nn_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Best model loaded (Epoch {checkpoint['epoch']+1})")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Val Accuracy: {checkpoint['val_accuracy']:.2f}%")
    print("="*70)

    return model

def predict_with_nn(model, domains, batch_size=1024):
    """
    Get predictions from neural network
    OPTIMIZED FOR: 96GB RAM (большие батчи)
    """
    dataset = DomainDataset(domains)

    # 🔥 Большой batch_size благодаря 96GB RAM
    loader = DataLoader(
        dataset,
        batch_size=batch_size,       # 🔥 1024 вместо 256
        shuffle=False,
        num_workers=16,              # 🔥 Много workers
        persistent_workers=True,
        prefetch_factor=4
    )

    model.eval()
    predictions = []

    print(f"🔮 Predicting with batch_size={batch_size}, workers=16...")

    with torch.no_grad():
        for batch_x in tqdm(loader, desc="Neural network predictions"):
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())

    return np.array(predictions)


# ======================================================
# Optimized Feature Extraction
# ======================================================

def make_X_parallel(df, fn, desc, n_jobs=-1):
    if n_jobs == -1:
        n_jobs = cpu_count()
    domains = [str(d) for d in df["domain"].tolist()]
    with Pool(n_jobs) as pool:
        results = list(tqdm(
            pool.imap(fn, domains, chunksize=1000),
            total=len(domains),
            desc=desc
        ))
    return np.array(results)

def cache_features(data, feature_func, cache_name, force_recompute=False):
    """
    Cache feature extraction with aggressive memory usage (96GB RAM)
    """
    cache_path = f"{cache_name}_ultra9.pkl"

    if os.path.exists(cache_path) and not force_recompute:
        print(f"✓ Loading cached features from {cache_path}...")
        with open(cache_path, 'rb') as f:
            features = pickle.load(f)
        print(f"  Shape: {features.shape}, Memory: {features.nbytes / 1e9:.2f} GB")
        return features

    print(f"🔄 Computing features (parallel processing)...")

    # 🔥 Очистка и конвертация доменов
    domains = data["domain"].fillna("").astype(str).tolist()

    # Удаляем пустые строки
    valid_indices = [i for i, d in enumerate(domains) if d.strip()]
    domains = [domains[i] for i in valid_indices]

    print(f"   Processing {len(domains)} valid domains...")

    # Разбиваем на чанки для параллельной обработки
    chunk_size = max(1, len(domains) // 20)
    chunks = [domains[i:i+chunk_size] for i in range(0, len(domains), chunk_size)]

    from multiprocessing import Pool

    results = []

    # Параллельно обрабатываем все чанки
    try:
        with Pool(processes=20) as pool:
            for chunk in tqdm(chunks, desc="Processing chunks"):
                try:
                    chunk_features = [feature_func(domain) for domain in chunk]
                    results.extend(chunk_features)
                except Exception as e:
                    print(f"\n⚠️  Error processing chunk: {e}")
                    # В случае ошибки добавляем нулевые признаки
                    num_features = len(feature_func(domains[0])) if domains else 28
                    results.extend([[0] * num_features for _ in chunk])
    except Exception as e:
        print(f"\n❌ Error in parallel processing: {e}")
        raise

    features = np.array(results, dtype=np.float32)

    print(f"✓ Features computed: {features.shape}, Memory: {features.nbytes / 1e9:.2f} GB")

    # Сохраняем в кэш
    with open(cache_path, 'wb') as f:
        pickle.dump(features, f, protocol=4)
    print(f"✓ Cached to {cache_path}")

    return features


# ======================================================
# Main Pipeline with PyTorch Neural Network
# ======================================================

def load_data():
    """Load training data"""
    print("Loading data...")
    #download_data()
    data = pd.read_csv("dga.csv")

    #"""" Optional: sample data for quick testing
    sampled_dfs = []
    for label, group in data.groupby("label"):
        sampled_group = group.sample(frac=0.02, random_state=42)
        sampled_dfs.append(sampled_group)

    data = pd.concat(sampled_dfs, ignore_index=True)

    print("Columns after sampling:", data.columns.tolist())
    print("Shape after sampling:", data.shape)
    # """

    return data


def train_models(data):
    """Train all models with parallel processing optimization"""

    print("="*70)
    print("DATA PREPARATION")
    print("="*70)

    # 🔥 ВАЖНО: Сначала разделяем данные, ПОТОМ извлекаем признаки
    print("\nSplitting data...")
    train, test = train_test_split(
        data, test_size=0.15, stratify=data["label"], random_state=42
    )
    train, val = train_test_split(
        train, test_size=0.15, stratify=train["label"], random_state=42
    )

    print(f"\nTrain: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Теперь извлекаем признаки для каждого набора отдельно
    print("\n" + "="*70)
    print("FEATURE EXTRACTION (PARALLEL)")
    print("="*70)

    # ============================================
    # PART 1: Traditional Models (LGB, LR, SVM)
    # ============================================

    print("\n" + "="*70)
    print("PART 1: TRADITIONAL ML MODELS")
    print("="*70)

    print("\n[1/5] Extracting tabular features...")
    X_train = cache_features(train, extract_features, "train_features_v7")
    X_val = cache_features(val, extract_features, "val_features_v7")
    X_test = cache_features(test, extract_features, "test_features_v7")

    y_train = train["label"].values
    y_val = val["label"].values
    y_test = test["label"].values

    print(f"Feature shape: {X_train.shape}")
    print(f"X_train samples: {len(X_train)}, y_train samples: {len(y_train)}")
    print(f"X_val samples: {len(X_val)}, y_val samples: {len(y_val)}")
    print(f"X_test samples: {len(X_test)}, y_test samples: {len(y_test)}")

    # Проверка консистентности
    assert len(X_train) == len(y_train), f"Train size mismatch: {len(X_train)} != {len(y_train)}"
    assert len(X_val) == len(y_val), f"Val size mismatch: {len(X_val)} != {len(y_val)}"
    assert len(X_test) == len(y_test), f"Test size mismatch: {len(X_test)} != {len(y_test)}"

    print("\n[2/5] Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=700,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight={0: 2.0, 1: 0.28},
        min_child_samples=30,
        reg_alpha=0.15,
        reg_lambda=0.15,
        random_state=42,
        n_jobs=-1,
        force_col_wise=True,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    p_val_lgb = lgb_model.predict_proba(X_val)[:, 1]
    p_test_lgb = lgb_model.predict_proba(X_test)[:, 1]

    print("\n[3/5] Vectorizing char-ngrams...")
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 6),
        min_df=20,
        max_features=24000,
        sublinear_tf=True,
        dtype=np.float32
    )
    X_ng_train = vectorizer.fit_transform(train["domain"].map(get_sld))
    X_ng_val = vectorizer.transform(val["domain"].map(get_sld))
    X_ng_test = vectorizer.transform(test["domain"].map(get_sld))

    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(
        max_iter=2000,
        C=0.75,
        class_weight={0: 2.0, 1: 0.28},
        n_jobs=-1,
        random_state=42
    )
    lr.fit(X_ng_train, y_train)
    p_val_lr = lr.predict_proba(X_ng_val)[:, 1]
    p_test_lr = lr.predict_proba(X_ng_test)[:, 1]

    print("\nTraining SVM...")
    svm = CalibratedClassifierCV(
        LinearSVC(
            C=0.55,
            class_weight={0: 2.5, 1: 0.16},
            max_iter=5000,
            random_state=42
        ),
        method="sigmoid",
        cv=3,
        n_jobs=-1
    )
    svm.fit(X_ng_train, y_train)
    p_val_svm = svm.predict_proba(X_ng_val)[:, 1]
    p_test_svm = svm.predict_proba(X_ng_test)[:, 1]

    # ============================================
    # PART 2: PyTorch Neural Network
    # ============================================

    print("\n" + "="*70)
    print("PART 2: PYTORCH DEEP LEARNING MODEL")
    print("="*70)

    print("\n[4/5] Preparing sequences for neural network...")
    train_domains = train["domain"].tolist()
    val_domains = val["domain"].tolist()
    test_domains = test["domain"].tolist()

    # Train neural network
    nn_model = train_neural_model(
        train_domains, y_train,
        val_domains, y_val,
        epochs=18,  # 🔥 Достаточно для CPU
        batch_size=512,  # 🔥 Большой batch для стабильности и скорости
        lr=0.001  # Стандартный learning rate
    )

    print("\n🔥 Getting neural network predictions...")
    p_val_nn = predict_with_nn(nn_model, val_domains)
    p_test_nn = predict_with_nn(nn_model, test_domains)

    print(f"NN predictions - Val: {p_val_nn.shape}, Test: {p_test_nn.shape}")

    # ============================================
    # PART 3: Meta-Model Ensemble
    # ============================================

    print("\n" + "="*70)
    print("PART 3: META-ENSEMBLE (4 MODELS)")
    print("="*70)

    print("\n[5/5] Extracting FP-Penalty features...")
    X_fp_val = cache_features(val, extract_fp_features, "fp_val_features_v6")
    X_fp_test = cache_features(test, extract_fp_features, "fp_test_features_v6")

    print("\nCreating meta-features with neural network...")

    # Averaged predictions
    p_val_avg = (p_val_lgb + p_val_lr + p_val_svm + p_val_nn) / 4
    p_test_avg = (p_test_lgb + p_test_lr + p_test_svm + p_test_nn) / 4

    # ML models average (without NN)
    p_val_ml_avg = (p_val_lgb + p_val_lr + p_val_svm) / 3
    p_test_ml_avg = (p_test_lgb + p_test_lr + p_test_svm) / 3

    # Conservative: all models agree
    p_val_conservative = np.maximum.reduce([p_val_lgb, p_val_lr, p_val_svm, p_val_nn])
    p_test_conservative = np.maximum.reduce([p_test_lgb, p_test_lr, p_test_svm, p_test_nn])

    # Aggressive: at least one model confident
    p_val_aggressive = np.minimum.reduce([p_val_lgb, p_val_lr, p_val_svm, p_val_nn])
    p_test_aggressive = np.minimum.reduce([p_test_lgb, p_test_lr, p_test_svm, p_test_nn])

    # NN disagreement with ML models
    p_val_nn_ml_diff = np.abs(p_val_nn - p_val_ml_avg)
    p_test_nn_ml_diff = np.abs(p_test_nn - p_test_ml_avg)

    # Variance across all models
    p_val_std = np.std([p_val_lgb, p_val_lr, p_val_svm, p_val_nn], axis=0)
    p_test_std = np.std([p_test_lgb, p_test_lr, p_test_svm, p_test_nn], axis=0)

    # Meta-feature matrix (31 features)
    X_meta_val = np.column_stack([
        p_val_lgb,
        p_val_lr,
        p_val_svm,
        p_val_nn,
        p_val_avg,
        p_val_ml_avg,
        p_val_conservative,
        p_val_aggressive,
        np.minimum(p_val_lr, p_val_svm),  # Text models agree on legit
        np.maximum(p_val_lgb, p_val_nn),  # Tree or NN confident DGA
        p_val_lgb * p_val_lr,  # LGB + LR both confident
        p_val_lgb * p_val_nn,  # LGB + NN both confident
        p_val_nn * p_val_lr,  # NN + LR both confident
        p_val_lgb * p_val_lr * p_val_svm * p_val_nn,  # All confident
        np.abs(p_val_lgb - p_val_lr),
        p_val_nn_ml_diff,
        p_val_std,
        X_fp_val  # 15 FP-protection features
    ])

    X_meta_test = np.column_stack([
        p_test_lgb,
        p_test_lr,
        p_test_svm,
        p_test_nn,
        p_test_avg,
        p_test_ml_avg,
        p_test_conservative,
        p_test_aggressive,
        np.minimum(p_test_lr, p_test_svm),
        np.maximum(p_test_lgb, p_test_nn),
        p_test_lgb * p_test_lr,
        p_test_lgb * p_test_nn,
        p_test_nn * p_test_lr,
        p_test_lgb * p_test_lr * p_test_svm * p_test_nn,
        np.abs(p_test_lgb - p_test_lr),
        p_test_nn_ml_diff,
        p_test_std,
        X_fp_test
    ])

    print(f"Meta-features shape: {X_meta_val.shape}")

    # Meta-model
    print("\nTraining Meta-LightGBM (Final Ensemble)...")
    meta_lgb = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=12,
        max_depth=7,
        min_child_samples=50,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=0.25,  # 🔥 Ultra FP protection
        reg_alpha=0.2,
        reg_lambda=0.2,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        force_col_wise=True,
        verbose=-1
    )
    meta_lgb.fit(X_meta_val, y_val)

    p_val_meta = meta_lgb.predict_proba(X_meta_val)[:, 1]
    p_test_meta = meta_lgb.predict_proba(X_meta_test)[:, 1]

    # ============================================
    # PART 4: Threshold Optimization
    # ============================================

    print("\n" + "=" * 70)
    print("PART 4: THRESHOLD OPTIMIZATION")
    print("=" * 70)

    print("\nFinding optimal threshold (ultra-fine search)...")
    best_thr, best_f = 0, 0

    # Coarse search
    for thr in np.linspace(0.25, 0.70, 800):
        preds = (p_val_meta >= thr).astype(int)
        f = fbeta_score(y_val, preds, beta=0.5)
        if f > best_f:
            best_f, best_thr = f, thr

    # Fine-tuning
    fine_range = np.linspace(max(0.2, best_thr - 0.08),
                             min(0.75, best_thr + 0.08), 2500)
    for thr in fine_range:
        preds = (p_val_meta >= thr).astype(int)
        f = fbeta_score(y_val, preds, beta=0.5)

        # Bonus for high precision (FP minimization)
        precision = precision_score(y_val, preds, zero_division=0)
        if precision > 0.985:
            f *= 1.003

        if f > best_f:
            best_f, best_thr = f, thr

    # ============================================
    # PART 5: Evaluation
    # ============================================

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"Best threshold: {best_thr:.4f}")
    print(f"Val F0.5: {best_f:.5f}")

    y_val_pred = (p_val_meta >= best_thr).astype(int)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    print(f"Val Precision: {val_precision:.5f} (🎯 FP minimization)")
    print(f"Val Recall: {val_recall:.5f}")

    # Individual model performance
    print("\n" + "-" * 70)
    print("Individual Model Performance on Validation:")
    print("-" * 70)

    for name, preds in [
        ("LightGBM", (p_val_lgb >= 0.5).astype(int)),
        ("LogReg", (p_val_lr >= 0.5).astype(int)),
        ("SVM", (p_val_svm >= 0.5).astype(int)),
        ("PyTorch NN", (p_val_nn >= 0.5).astype(int))
    ]:
        f05 = fbeta_score(y_val, preds, beta=0.5)
        prec = precision_score(y_val, preds)
        rec = recall_score(y_val, preds)
        print(f"{name:12s} - F0.5: {f05:.5f}, Precision: {prec:.5f}, Recall: {rec:.5f}")

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    y_test_pred = (p_test_meta >= best_thr).astype(int)

    test_f05 = fbeta_score(y_test, y_test_pred, beta=0.5)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)

    print(f"\n🎯 Test F0.5: {test_f05:.5f}")
    print(f"   Test Precision: {test_precision:.5f}")
    print(f"   Test Recall: {test_recall:.5f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_test_pred, digits=4))

    fp_count = np.sum((y_test == 0) & (y_test_pred == 1))
    fn_count = np.sum((y_test == 1) & (y_test_pred == 0))
    print(f"\nError Analysis:")
    print(f"  False Positives: {fp_count} (legit → DGA)")
    print(f"  False Negatives: {fn_count} (DGA → legit)")
    print(f"  FP/FN Ratio: {fp_count / max(1, fn_count):.2f}")
    print(f"  Total errors: {fp_count + fn_count} ({(fp_count + fn_count) / len(y_test) * 100:.2f}%)")

    return lgb_model, lr, svm, vectorizer, nn_model, meta_lgb, best_thr


def create_submission(lgb_model, lr, svm, vectorizer, nn_model, meta_lgb, best_thr):
    """Create submission file from test data"""
    if not os.path.exists('test.csv'):
        print("\nWarning: test.csv not found. Skipping submission creation.")
        return

    print("\n" + "=" * 70)
    print("CREATING SUBMISSION")
    print("=" * 70)

    data_test = pd.read_csv("test.csv")
    print(f"Test set size: {len(data_test)}")

    # 🔥 Очистка данных от NaN
    print("\nCleaning test data...")
    original_size = len(data_test)

    # Проверяем наличие NaN в колонке domain
    nan_count = data_test['domain'].isna().sum()
    if nan_count > 0:
        print(f"⚠️  Found {nan_count} NaN values in 'domain' column")
        # Заменяем NaN на пустую строку или удаляем
        data_test = data_test[data_test['domain'].notna()].copy()
        print(f"   Removed {original_size - len(data_test)} invalid rows")

    # Конвертируем все домены в строки
    data_test['domain'] = data_test['domain'].astype(str)

    # Удаляем пустые строки
    data_test = data_test[data_test['domain'].str.strip() != ''].copy()

    if len(data_test) < original_size:
        print(f"✓ Cleaned test set size: {len(data_test)} (removed {original_size - len(data_test)} invalid entries)")


    # Extract features
    print("\n[1/4] Extracting tabular features...")
    X_test_features = cache_features(data_test, extract_features, "submission_features_v6")

    print("[2/4] Extracting FP-penalty features...")
    X_fp_test = cache_features(data_test, extract_fp_features, "submission_fp_features_v6")

    print("[3/4] Vectorizing char-ngrams...")
    X_ngram_test = vectorizer.transform(
        data_test["domain"].map(get_sld)
    )

    print("[4/4] Preparing sequences for neural network...")
    test_domains = data_test["domain"].tolist()

    # Base model predictions
    print("\nPredicting with base models...")
    p_test_lgb = lgb_model.predict_proba(X_test_features)[:, 1]
    p_test_lr = lr.predict_proba(X_ngram_test)[:, 1]
    p_test_svm = svm.predict_proba(X_ngram_test)[:, 1]

    print("Predicting with PyTorch neural network...")
    p_test_nn = predict_with_nn(nn_model, test_domains)

    # Create meta-features
    print("Creating meta-features...")
    p_test_avg = (p_test_lgb + p_test_lr + p_test_svm + p_test_nn) / 4
    p_test_ml_avg = (p_test_lgb + p_test_lr + p_test_svm) / 3
    p_test_conservative = np.maximum.reduce([p_test_lgb, p_test_lr, p_test_svm, p_test_nn])
    p_test_aggressive = np.minimum.reduce([p_test_lgb, p_test_lr, p_test_svm, p_test_nn])
    p_test_nn_ml_diff = np.abs(p_test_nn - p_test_ml_avg)

    X_meta_test = np.column_stack([
        p_test_lgb,
        p_test_lr,
        p_test_svm,
        p_test_nn,
        p_test_avg,
        p_test_ml_avg,
        p_test_conservative,
        p_test_aggressive,
        np.minimum(p_test_lr, p_test_svm),
        np.maximum(p_test_lgb, p_test_nn),
        p_test_lgb * p_test_lr,
        p_test_lgb * p_test_nn,
        p_test_nn * p_test_lr,
        p_test_lgb * p_test_lr * p_test_svm * p_test_nn,
        np.abs(p_test_lgb - p_test_lr),
        p_test_nn_ml_diff,
        np.std([p_test_lgb, p_test_lr, p_test_svm, p_test_nn], axis=0),
        X_fp_test
    ])

    # Meta-model prediction
    print("Final prediction with meta-ensemble...")
    p_meta_test = meta_lgb.predict_proba(X_meta_test)[:, 1]

    # Final decision
    data_test["label"] = (p_meta_test >= best_thr).astype(int)

    # Save submission
    data_test[["id", "label"]].to_csv("submission_v6_pytorch.csv", index=False)

    # Statistics
    dga_count = data_test['label'].sum()
    dga_ratio = dga_count / len(data_test) * 100

    print(f"\n✓ Submission Statistics:")
    print(f"  Total domains: {len(data_test)}")
    print(f"  Predicted DGA: {dga_count} ({dga_ratio:.2f}%)")
    print(f"  Predicted Legitimate: {len(data_test) - dga_count} ({100 - dga_ratio:.2f}%)")
    print(f"  Threshold used: {best_thr:.4f}")

    # Model agreement analysis
    print(f"\n  Model predictions (avg probabilities):")
    print(f"    LightGBM: {p_test_lgb.mean():.4f}")
    print(f"    LogReg: {p_test_lr.mean():.4f}")
    print(f"    SVM: {p_test_svm.mean():.4f}")
    print(f"    PyTorch NN: {p_test_nn.mean():.4f}")
    print(f"    Meta-Ensemble: {p_meta_test.mean():.4f}")

    print(f"\n✓ Submission saved to submission_v6_pytorch.csv")


def main():
    """Main execution function"""
    start = time.time()

    print("=" * 70)
    print("DGA DETECTION PIPELINE v6.0 - PYTORCH NEURAL ENSEMBLE")
    print("=" * 70)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\n🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠️  No GPU detected, using CPU (training will be slower)")

    # Load data
    data = load_data()
    print("\nData preview:")
    print(data.head())
    print(f"\nDataset shape: {data.shape}")
    print(f"Label distribution:\n{data['label'].value_counts()}")

    # Train models
    models = train_models(data)

    # Create submission (if test.csv exists)
    create_submission(*models)

    print("\n" + "=" * 70)
    elapsed = time.time() - start
    print(f"PIPELINE COMPLETED SUCCESSFULLY ✓")
    print(f"Total execution time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    print("=" * 70)


if __name__ == "__main__":
    main()