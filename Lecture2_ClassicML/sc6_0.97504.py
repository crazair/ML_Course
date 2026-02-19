import os
import sys
import subprocess
from math import log2
import time
import pickle
import hashlib
import re

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
from multiprocessing import Pool, cpu_count

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
    'store','retail','shop','auto','car','travel','booking',
    'food','health','care','life','free','best','top','new'
])

VOWELS = set("aeiou")

COMMON_TLDS = set([
    'com','org','net','edu','gov','io','co','ai','app','dev',
    'tech','online','store','shop','site','web','cloud','info',
    'biz','us','uk','de','jp','cn','ru','br','in','au'
])


# ======================================================
# Feature Engineering Functions
# ======================================================

def contains_common_word(name):
    """Check if domain contains common words"""
    return int(any(w in name for w in COMMON_WORDS))


def shannon_entropy(s):
    """Calculate Shannon entropy of a string"""
    if not s:
        return 0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * log2(p) for p in probs)


def digit_ratio(name):
    """Calculate ratio of digits in domain name"""
    return sum(c.isdigit() for c in name) / max(1, len(name))


def vowel_ratio(name):
    """Calculate ratio of vowels in domain name"""
    letters = sum(c.isalpha() for c in name)
    vowels = sum(c in VOWELS for c in name)
    return vowels / max(1, letters)


def clean_entropy(name):
    """Calculate entropy of alphabetic characters only"""
    clean = ''.join(c for c in name if c.isalpha())
    return shannon_entropy(clean)


def max_consonant_run(name):
    """Find maximum consecutive consonants"""
    run = max_run = 0
    for c in name:
        if c.isalpha() and c not in VOWELS:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def char_diversity(name):
    """Calculate character diversity"""
    return len(set(name)) / max(1, len(name))


def bigram_diversity(name):
    """Calculate bigram diversity"""
    if len(name) < 2:
        return 0
    b = [name[i:i+2] for i in range(len(name)-1)]
    return len(set(b)) / len(b)


def type_switch_ratio(name):
    """Calculate ratio of character type switches"""
    def t(c):
        if c.isdigit():
            return "d"
        if c.isalpha():
            return "a"
        return "o"

    switches = sum(t(name[i]) != t(name[i-1]) for i in range(1, len(name)))
    return switches / max(1, len(name)-1)


# ====== НОВОЕ: Дополнительные признаки ======

def has_valid_tld(domain):
    """Check if domain has a common TLD"""
    parts = domain.split('.')
    if len(parts) < 2:
        return 0
    return int(parts[-1].lower() in COMMON_TLDS)


def repeated_pattern_score(name):
    """Detect repeated patterns (common in DGA)"""
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
    """Calculate pronounceability (DGA domains are often unpronounceable)"""
    if not name:
        return 0

    # Count valid consonant-vowel patterns
    valid_patterns = 0
    for i in range(len(name) - 1):
        c1, c2 = name[i], name[i+1]
        if c1.isalpha() and c2.isalpha():
            # CV or VC patterns are pronounceable
            if (c1 in VOWELS and c2 not in VOWELS) or (c1 not in VOWELS and c2 in VOWELS):
                valid_patterns += 1

    return valid_patterns / max(1, len(name) - 1)


def rare_char_ratio(name):
    """Ratio of rare characters (q, x, z, etc.)"""
    rare_chars = set('qxzjkv')
    return sum(c in rare_chars for c in name.lower()) / max(1, len(name))


def trigram_diversity(name):
    """Calculate trigram diversity"""
    if len(name) < 3:
        return 0
    tg = [name[i:i+3] for i in range(len(name)-2)]
    return len(set(tg)) / len(tg) if tg else 0


def extract_features(domain):
    """Extract tabular features from domain (РАСШИРЕНО)"""
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
        # === НОВЫЕ ПРИЗНАКИ ===
        has_valid_tld(domain),
        repeated_pattern_score(name),
        pronounceability_score(name),
        rare_char_ratio(name),
        trigram_diversity(name),
        int(len(name) > 15),  # Very long domains often DGA
        int('.' in domain),   # Has TLD separator
    ]


def extract_fp_features(domain):
    """Extract FP-Penalty features from domain (РАСШИРЕНО)"""
    name = domain.split('.')[0].lower()

    return [
        contains_common_word(name),
        int(len(name) <= 6),
        vowel_ratio(name),
        digit_ratio(name),
        shannon_entropy(name),
        max_consonant_run(name),
        # === НОВЫЕ ПРИЗНАКИ ===
        has_valid_tld(domain),
        pronounceability_score(name),
        int(len(name) <= 10 and vowel_ratio(name) > 0.3),  # Short + pronounceable
    ]


def get_sld(d):
    """Get second-level domain"""
    return str(d).split('.')[0].lower()


# ======================================================
# Optimized Feature Extraction
# ======================================================

def make_X_parallel(df, fn, desc, n_jobs=-1):
    """Create feature matrix using parallel processing"""
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


def cache_features(df, fn, cache_name):
    """Cache feature extraction results"""
    cache_file = f"{cache_name}.pkl"
    cache_meta = f"{cache_name}_meta.txt"

    # Создайте хэш для проверки изменений данных
    data_hash = hashlib.md5(
        pd.util.hash_pandas_object(df['domain']).values
    ).hexdigest()

    if os.path.exists(cache_file) and os.path.exists(cache_meta):
        with open(cache_meta, 'r') as f:
            cached_hash = f.read().strip()
            if cached_hash == data_hash:
                print(f"✓ Loading cached {cache_name}...")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

    # Вычислить признаки
    print(f"Computing {cache_name}...")
    features = make_X_parallel(df, fn, f"{cache_name}", n_jobs=-1)

    # Сохранить кэш
    with open(cache_file, 'wb') as f:
        pickle.dump(features, f)
    with open(cache_meta, 'w') as f:
        f.write(data_hash)

    return features


# ======================================================
# Main Pipeline (УЛУЧШЕНО)
# ======================================================
def load_data():
    """Load training data"""
    print("Loading data...")
    #download_data()
    data = pd.read_csv("dga.csv")

    #"""" Optional: sample data for quick testing
    sampled_dfs = []
    for label, group in data.groupby("label"):
        sampled_group = group.sample(frac=0.01, random_state=42)
        sampled_dfs.append(sampled_group)

    data = pd.concat(sampled_dfs, ignore_index=True)

    print("Columns after sampling:", data.columns.tolist())
    print("Shape after sampling:", data.shape)
    # """

    return data


def train_models(data):
    """Train all models in the pipeline - OPTIMIZED VERSION"""
    print("\n" + "="*60)
    print("TRAINING PIPELINE (OPTIMIZED v2.0 - F0.5 > 0.98)")
    print("="*60)

    # Train/Val/Test split
    print("\nSplitting data...")
    train, test = train_test_split(
        data, test_size=0.15, stratify=data["label"], random_state=42
    )

    train, val = train_test_split(
        train, test_size=0.15, stratify=train["label"], random_state=42
    )

    # Extract features WITH CACHING (с расширенными признаками)
    print("\n[1/3] Extracting enhanced features (with caching)...")
    X_train = cache_features(train, extract_features, "train_features_v2")
    X_val = cache_features(val, extract_features, "val_features_v2")
    X_test = cache_features(test, extract_features, "test_features_v2")

    y_train = train["label"].values
    y_val = val["label"].values
    y_test = test["label"].values

    # === ИЗМЕНЕНИЕ 1: Оптимизированные веса для минимизации FP ===
    print("\n[2/3] Training LightGBM (FP-optimized)...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=600,           # Увеличено для лучшей сходимости
        learning_rate=0.04,         # Снижено для стабильности
        num_leaves=31,
        max_depth=8,                # Увеличена глубина
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight={0: 1.5, 1: 0.35},  # УСИЛЕН вес класса 0 (легитимные)
        min_child_samples=25,       # Защита от переобучения
        reg_alpha=0.1,              # L1 регуляризация
        reg_lambda=0.1,             # L2 регуляризация
        random_state=42,
        n_jobs=-1,
        force_col_wise=True,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)

    p_val_lgb = lgb_model.predict_proba(X_val)[:, 1]
    p_test_lgb = lgb_model.predict_proba(X_test)[:, 1]

    # === ИЗМЕНЕНИЕ 2: Улучшенная n-gram модель ===
    print("\n[3/3] Vectorizing char-ngrams (enhanced)...")
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 6),        # РАСШИРЕН диапазон (было 3-5)
        min_df=25,                 # Снижен порог (было 30)
        max_features=20000,        # УВЕЛИЧЕНО (было 15000)
        sublinear_tf=True,
        dtype=np.float32
    )

    X_ng_train = vectorizer.fit_transform(train["domain"].map(get_sld))
    X_ng_val = vectorizer.transform(val["domain"].map(get_sld))
    X_ng_test = vectorizer.transform(test["domain"].map(get_sld))

    # Train Logistic Regression (FP-optimized)
    print("\nTraining Logistic Regression (FP-optimized)...")
    lr = LogisticRegression(
        max_iter=2000,
        C=0.8,                      # Добавлена регуляризация
        class_weight={0: 1.5, 1: 0.35},  # УСИЛЕН вес класса 0
        n_jobs=-1,
        random_state=42
    )
    lr.fit(X_ng_train, y_train)

    p_val_lr = lr.predict_proba(X_ng_val)[:, 1]
    p_test_lr = lr.predict_proba(X_ng_test)[:, 1]

    # Train SVM (FP-optimized)
    print("\nTraining SVM (FP-optimized)...")
    svm = CalibratedClassifierCV(
        LinearSVC(
            C=0.6,                  # Увеличена регуляризация
            class_weight={0: 1.8, 1: 0.2},  # УСИЛЕН вес класса 0
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

    # FP-Penalty features WITH CACHING (расширенные)
    print("\nExtracting FP-Penalty features (enhanced)...")
    X_fp_val = cache_features(val, extract_fp_features, "fp_val_features_v2")
    X_fp_test = cache_features(test, extract_fp_features, "fp_test_features_v2")

    # === ИЗМЕНЕНИЕ 3: Расширенные мета-фичи ===
    print("\nCreating enhanced meta-features...")
    X_meta_val = np.column_stack([
        p_val_lgb,
        p_val_lr,
        p_val_svm,
        np.minimum(p_val_lr, p_val_svm),  # Min (консервативный)
        np.maximum(p_val_lgb, p_val_lr),  # Max (агрессивный)
        (p_val_lgb + p_val_lr + p_val_svm) / 3,  # Среднее
        p_val_lgb * p_val_lr,  # Произведение (оба должны быть уверены)
        np.abs(p_val_lgb - p_val_lr),  # Разница (мера несогласия)
        X_fp_val
    ])

    X_meta_test = np.column_stack([
        p_test_lgb,
        p_test_lr,
        p_test_svm,
        np.minimum(p_test_lr, p_test_svm),
        np.maximum(p_test_lgb, p_test_lr),
        (p_test_lgb + p_test_lr + p_test_svm) / 3,
        p_test_lgb * p_test_lr,
        np.abs(p_test_lgb - p_test_lr),
        X_fp_test
    ])

    # Train Meta-LightGBM (FP-optimized)
    print("\nTraining Meta-LightGBM (FP-optimized)...")
    meta_lgb = lgb.LGBMClassifier(
        n_estimators=500,  # Увеличено
        learning_rate=0.04,  # Снижено для лучшей точности
        num_leaves=12,  # Снижено для уменьшения переобучения
        max_depth=6,  # Увеличена глубина
        min_child_samples=40,  # Защита от переобучения
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=0.3,  # УСИЛЕНА защита от FP (было 0.35)
        reg_alpha=0.15,  # L1 регуляризация
        reg_lambda=0.15,  # L2 регуляризация
        objective="binary",
        random_state=42,
        n_jobs=-1,
        force_col_wise=True,
        verbose=-1
    )
    meta_lgb.fit(X_meta_val, y_val)

    p_val_meta = meta_lgb.predict_proba(X_meta_val)[:, 1]
    p_test_meta = meta_lgb.predict_proba(X_meta_test)[:, 1]

    # Find best threshold (более агрессивный поиск)
    print("\nFinding best threshold for F0.5 (fine-tuned search)...")
    best_thr, best_f = 0, 0

    # Сначала грубый поиск
    for thr in np.linspace(0.2, 0.65, 500):
        preds = (p_val_meta >= thr).astype(int)
        f = fbeta_score(y_val, preds, beta=0.5)
        if f > best_f:
            best_f, best_thr = f, thr

    # Затем тонкая настройка вокруг лучшего порога
    fine_range = np.linspace(max(0.15, best_thr - 0.05),
                             min(0.7, best_thr + 0.05), 1000)
    for thr in fine_range:
        preds = (p_val_meta >= thr).astype(int)
        f = fbeta_score(y_val, preds, beta=0.5)
        if f > best_f:
            best_f, best_thr = f, thr

    print(f"\n{'=' * 60}")
    print(f"VALIDATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Best threshold: {best_thr:.4f}")
    print(f"Val F0.5: {best_f:.5f}")

    # Дополнительная валидация
    y_val_pred = (p_val_meta >= best_thr).astype(int)
    from sklearn.metrics import precision_score, recall_score
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    print(f"Val Precision: {precision:.5f} (важно для минимизации FP)")
    print(f"Val Recall: {recall:.5f}")

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)
    y_test_pred = (p_test_meta >= best_thr).astype(int)

    test_f05 = fbeta_score(y_test, y_test_pred, beta=0.5)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)

    print(f"\n🎯 Test F0.5: {test_f05:.5f}")
    print(f"   Test Precision: {test_precision:.5f}")
    print(f"   Test Recall: {test_recall:.5f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_test_pred, digits=4))

    # Анализ ошибок
    fp_count = np.sum((y_test == 0) & (y_test_pred == 1))
    fn_count = np.sum((y_test == 1) & (y_test_pred == 0))
    print(f"\nError Analysis:")
    print(f"  False Positives: {fp_count} (легитимные помечены как DGA)")
    print(f"  False Negatives: {fn_count} (DGA пропущены)")
    print(f"  FP/FN Ratio: {fp_count / max(1, fn_count):.2f}")

    return lgb_model, lr, svm, vectorizer, meta_lgb, best_thr


def create_submission(lgb_model, lr, svm, vectorizer, meta_lgb, best_thr):
    """Create submission file from test data"""
    if not os.path.exists('test.csv'):
        print("\nWarning: test.csv not found. Skipping submission creation.")
        return

    print("\n" + "=" * 60)
    print("CREATING SUBMISSION")
    print("=" * 60)

    data_test = pd.read_csv("test.csv")

    # Tabular features with caching (v2 с новыми признаками)
    X_test_features = cache_features(data_test, extract_features, "submission_features_v2")

    # FP-Penalty features with caching (v2)
    X_fp_test = cache_features(data_test, extract_fp_features, "submission_fp_features_v2")

    # Char-ngrams
    X_ngram_test = vectorizer.transform(
        data_test["domain"].map(get_sld)
    )

    # Base model probabilities
    print("Predicting with base models...")
    p_test_lgb = lgb_model.predict_proba(X_test_features)[:, 1]
    p_test_lr = lr.predict_proba(X_ngram_test)[:, 1]
    p_test_svm = svm.predict_proba(X_ngram_test)[:, 1]

    # Enhanced meta-features
    print("Creating meta-features...")
    X_meta_test = np.column_stack([
        p_test_lgb,
        p_test_lr,
        p_test_svm,
        np.minimum(p_test_lr, p_test_svm),
        np.maximum(p_test_lgb, p_test_lr),
        (p_test_lgb + p_test_lr + p_test_svm) / 3,
        p_test_lgb * p_test_lr,
        np.abs(p_test_lgb - p_test_lr),
        X_fp_test
    ])

    # Meta-model prediction
    print("Final prediction with meta-model...")
    p_meta_test = meta_lgb.predict_proba(X_meta_test)[:, 1]

    # Final decision with optimized threshold
    data_test["label"] = (p_meta_test >= best_thr).astype(int)

    # Save submission
    data_test[["id", "label"]].to_csv("submission_sr6.csv", index=False)

    # Статистика
    dga_count = data_test['label'].sum()
    dga_ratio = dga_count / len(data_test) * 100

    print(f"\n✓ Submission Statistics:")
    print(f"  Total domains: {len(data_test)}")
    print(f"  Predicted DGA: {dga_count} ({dga_ratio:.2f}%)")
    print(f"  Predicted Legitimate: {len(data_test) - dga_count} ({100 - dga_ratio:.2f}%)")
    print(f"  Threshold used: {best_thr:.4f}")
    print("\n✓ Submission saved to submission.csv")


def main():
    start = time.time()
    """Main execution function"""
    print("=" * 60)
    print("DGA DETECTION PIPELINE")
    print("=" * 60)

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

    print("\n" + "=" * 60)
    elapsed = time.time() - start
    print( f"PIPELINE COMPLETED SUCCESSFULLY ⏱ Total execution time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    print("=" * 60)


if __name__ == "__main__":
    main()