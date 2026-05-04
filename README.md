---

**Azərbaycan ASR — Whisper LoRA ilə Fine-Tuning**

Aşağı resurslu dillər üçün ASR • Parametr-effektiv öyrətmə • İstehsalat səviyyəli boru kəməri

### İcra Xülasəsi

Azərbaycan dili üçün istehsalat yönümlü Avtomatik Nitqi Tanınma (ASR) sistemi. 

Morfoloji cəhətdən zəngin və aşağı resurslu türk dili olan Azərbaycan dili üçün `LocalDoc/azerbaijani-whisper-small` modeli əsasında qurulmuşdur. Model **LoRA** (Low-Rank Adaptation) adapterləri ilə fine-tune edilmişdir və cəmi **1%-dən az** öyrədilə bilən parametr istifadə olunur.

**Google Colab T4 (15GB VRAM)** kimi məhdud resurslu mühitlər üçün optimallaşdırılmışdır. Tam monitorinq, qiymətləndirmə və deployment hazırdır.

**Əsas Nailiyyət:** Yalnız 200 treninq nümunəsi ilə validasiya setində yaxşı nəticə əldə edilmişdir. Overfitting yoxdur. Əsas problem məlumat həcmi ilə bağlıdır, metodologiya ilə deyil.

### Problem

Azərbaycan dilində ASR üçün əsas çətinliklər:

Çətinlik                        | Təsir                                   | Nümunə
--------------------------------|-----------------------------------------|---------------------------
Agglutinativ morfologiya        | Sözə 6+ şəkilçi əlavəsi, yüksək qeyri-müəyyənlik | planlaşdırılırdı (6 morfem)
Qeyri-latın fonemləri           | Nadir hərflər səbəbindən səhv            | ə, ö, ü, ğ, ç, ş
Yetərsiz təmsil olunan sözlər   | Xüsusi isimlər və alınma sözlər səhv çıxır | Zəngəzur → Zəngüzül
Sait uyğunluğu qaydaları        | Şəkilçi variantları tokenizeri çaşdırır | ərazisindən → ərazisində

**İstehsalat hədəfi:** WER < 10%

### Arxitektura və Dizayn Qərarları

**Model Seçimi:**

Model                | Parametrlər | VRAM     | Qərar
---------------------|-------------|----------|--------------------
Whisper Large-v3     | 1550M       | 32GB+    | ❌ T4 üçün çoxdur
Whisper Medium       | 769M        | 18GB+    | ❌ Colab-da OOM
Whisper Small        | 244M        | ~6GB     | ✅ Optimal seçim
Whisper Tiny         | 39M         | 3GB      | ❌ Aşağı dəqiqlik

**LoRA Konfiqurasiyası:**

Parametr            | Dəyər               | Əsaslandırma
--------------------|---------------------|--------------------------------
r (rank)            | 16                  | 200 nümunə üçün ən yaxşı balans
lora_alpha          | 32                  | Daha güclü adaptasiya
lora_dropout        | 0.05                | Yüngül tənzimləmə
target_modules      | q_proj, v_proj      | Yalnız diqqət qatları
Öyrədilən parametrlər | 1.2M / 244M (0.49%) | Dilə uyğunlaşma üçün kifayətdir

**Treninq Strategiyası:**
- Avadanlıq: Google Colab T4 (15GB VRAM)
- Batch ölçüsü: 4 (gradient accumulation ilə effektiv 8)
- Dəqiqlik: FP16 mixed-precision
- Addım sayı: 200
- Qiymətləndirmə: Hər 20 addımdan bir
- Checkpoint: Ən yaxşı validation WER

### Nəticələr və Təhlil

**Test Nəticələri:**

Model                     | WER (%) | CER (%) | Status
--------------------------|---------|---------|--------------------
Baseline (LocalDoc)       | 22.65   | 7.00    | —
Fine-tuned (LoRA)         | 23.06   | 7.65    | Bir qədər yüksək
İstehsalat Hədəfi         | <10.0   | <5.0    | —

**Treninq Dinamikası:**

Metrik              | İlkin   | Son    | Dəyişiklik | Şərh
--------------------|---------|--------|------------|-------------------
Treninq itkisi      | 2.849   | 0.872  | ↓69%       | Sürətli yaxşılaşma
Validasiya itkisi   | 1.995   | 1.003  | ↓50%       | Stabil
Validasiya WER      | —       | —      | Yaxşılaşdı | Müsbət trend
Validasiya CER      | —       | —      | Yaxşılaşdı | Müsbət trend

Overfitting yoxdur (fərq 0.131 < 0.3).

**Səhv Növlərinin Paylanması:**
- Fonetik qarışıqlıq       : 35%
- Leksik əvəzləmə         : 25%
- Morfoloji səhvlər       : 20%
- Xüsusi isim səhvləri    : 15%
- Tokenizasiya səhvləri   : 5%

**Audio Şərtlərinə Görə Nəticə:**

Audio Şərti                          | WER Aralığı | Status             | Tövsiyə
-------------------------------------|-------------|--------------------|--------------------
Qısa və təmiz audio (5-10 söz)       | 0-5%        | İstehsalata hazır  | Birbaşa istifadə
Ədəbi dil                            | 0-5%        | İstehsalata hazır  | Birbaşa istifadə
Uzun və mürəkkəb cümlələr            | 50-85%      | Orta               | Chunking istifadə et
Nadir termin və xüsusi isimlər       | 50-100%     | Orta               | Entity LM əlavə et
Səsli / dialektli nitq               | 60-80%      | Zəif               | Data augmentation

### Tez Başlama

**Tələblər:**
- Python 3.10+
- CUDA 11.8+ (GPU tövsiyə olunur)
- Ən azı 8GB RAM (15GB VRAM arzuolunandır)

**Quraşdırma:**
1. `git clone https://github.com/rafiveyisov/azerbaijani-asr-whisper.git`
2. `cd azerbaijani-asr-whisper`
3. `pip install -r requirements.txt`

**Dataset strukturu:**
```
data/
├── clips/           # 16kHz mono .wav faylları
├── train.tsv
├── dev.tsv
└── test.tsv
```

### Repozitoriya Strukturu

azerbaijani-asr-whisper/
├── README.md
├── requirements.txt
├── part_a/           # Məlumat analizi
├── part_b/           # Treninq
├── part_c/           # Qiymətləndirmə
├── results/          # Qrafiklər və nəticələr
├── report.pdf
└── .gitignore

### İstehsalat Yol Xəritəsi

- Qısamüddətli: ONNX + INT8 kvantlaşdırma, FastAPI deployment
- Ortamüddətli: İstifadəçi düzəlişləri ilə data flywheel
- Uzunmüddətli: Böyük dataset + daha böyük model + xarici dil modeli

**Hədəf:** Kifayət qədər məlumatla 8–12% WER.

### Məlum Məhdudiyyətlər və Həllər

Məhdudiyyət                    | Həll yolu
-------------------------------|-----------------------------------
Kiçik treninq dataseti         | Məlumatı artırmaq (ən prioritet)
Audio augmentation olmaması    | Arxa plan səs əlavə etmək
Xarici dil modeli olmaması     | KenLM və ya Neural LM inteqrasiya etmək
Xüsusi isim səhvləri           | NER post-processing əlavə etmək

### Sitasyon və Lisenziya

**Lisenziya:** MIT License

**Dataset:** Mozilla Common Voice (CC0-1.0)

---

Bu versiya tam Azərbaycanca və səliqəlidir. Word-ə yapışdırmaq üçün çox uyğundur. Əlavə dəyişiklik istəyirsənsə (məsələn, başlıqları dəyişmək, daha qısa etmək və s.) de.
