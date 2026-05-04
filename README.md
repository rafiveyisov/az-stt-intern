# 🎙️ Azərbaycan Nitq Tanıma Sistemi (ASR)

---

## 📌 Qısa İzahat

Bu layihə Azərbaycan dili üçün Avtomatik Nitq Tanıma (ASR) sisteminin qurulmasını nümayiş etdirir. Baza model olaraq `LocalDoc/azerbaijani-whisper-small` seçilmiş, Mozilla Common Voice 25.0 Azerbaijani dataset-i üzərində qiymətləndirilmişdir. Hissə B-də həmin modelə LoRA (Low-Rank Adaptation) ilə parameter-efficient fine-tuning tətbiq edilmişdir. Məqsəd pipeline-ı texniki cəhətdən düzgün qurmaq, overfitting-dən qaçmaq və nəticələri müqayisə etməkdir.

---

## 🧠 İstifadə Olunan Model və Parametrlər

**Baza Model:** `LocalDoc/azerbaijani-whisper-small`
- Encoder: 12 transformer layers
- Decoder: 12 transformer layers
- Hidden size: 768
- Ümumi parametrlər: ~244M

**LoRA Konfiqurasiyası:**
- r (rank): 16
- lora_alpha: 32
- lora_dropout: 0.05
- target_modules: q_proj, v_proj
- Öyrənilən parametrlər: ~1.2M

**Təlim Parametrləri:**
- Learning rate: 1e-4
- Batch size: 4
- Max steps: 150
- Eval/Save steps: 20
- Train nümunə sayı: 126 (Part A) / 200 (Part B)
- Precision: FP16
- Best model metric: WER

---

## 📊 WER/CER Nəticələri

### Hissə A — Baza Model (Test Set)

<table>
<tr><th>Metrik</th><th>Dəyər</th></tr>
<tr><td>Ortalama WER</td><td>22.65%</td></tr>
<tr><td>Ortalama CER</td><td>7.00%</td></tr>
</table>

### Hissə B — Fine-Tuned Model Müqayisəsi (Test Set)

<table>
<tr><th>Model</th><th>WER (%)</th><th>CER (%)</th></tr>
<tr><td>Baza Model</td><td>22.65</td><td>7.00</td></tr>
<tr><td>Fine-tuned Model</td><td>23.06</td><td>7.65</td></tr>
<tr><td><b>Fərq</b></td><td><b>+0.41</b></td><td><b>+0.65</b></td></tr>
</table>

### Validation Gedişatı (Təlim Zamanı)

<table>
<tr><th>Metrik</th><th>Başlanğıc</th><th>Son</th><th>Ən Yaxşı</th></tr>
<tr><td>Training Loss</td><td>2.849</td><td>0.872</td><td>—</td></tr>
<tr><td>Validation Loss</td><td>1.995</td><td>1.003</td><td>—</td></tr>
<tr><td>Validation WER</td><td>—</td><td>—</td><td>29.7%</td></tr>
<tr><td>Validation CER</td><td>—</td><td>—</td><td>9.0%</td></tr>
</table>

**Overfitting Analizi:** Δ (Val Loss − Train Loss) = 0.131. Hədd < 0.3 olduğu üçün overfitting yoxdur. ✅

---

## 📈 Təlim Qrafiki

![Training Progress](results/training_progress.png)

*Training Loss (mavi), Validation Loss (qırmızı), Validation WER (narıncı) və CER (bənövşəyi). Qırmızı nöqtə ən yaxşı checkpoint-i göstərir.*

---

## ⚡ Kodu İşə Salmaq üçün Addımlar

**1. Repository-i klonlayın:**
```bash
git clone https://github.com/rafiveyisov/az-stt-intern.git
cd az-stt-intern
```

**2. Asılılıqları qurun:**
```bash
pip install -r requirements.txt
```

**3. Dataset-i yerləşdirin:**
Mozilla Common Voice dataset-ini `../az/` qovluğuna çıxarın:
```
../az/
├── clips/
├── train.tsv
├── dev.tsv
└── test.tsv
```

**4. Hissə A — Baza model ilə işləmək:**
```bash
python part_a/asr_code.py
```
Test set-dən 126 nümunə üzərində WER/CER hesablayır, ən yaxşı və ən pis 5 nümunəni çap edir.

**5. Hissə B — Fine-tuning və müqayisə:**
```bash
python part_b/fine_tuning.py
```
Modeli 150 addım fine-tune edir, hər 20 addımdan bir validation aparır, ən yaxşı checkpoint-i saxlayır, baza və fine-tuned modelləri test set-də müqayisə edir. Nəticələr `results/` qovluğuna yazılır.

**6. Hesabat:**
`report.pdf` faylında çətinliklər, nəticələrin təhlili və gələcək addımlar təqdim olunur.

---

## 🔬 Fine-Tuning Nəticəsinin Müqayisəsi

Fine-tuned model test set-də baza modellə minimal fərq göstərir (+0.41% WER). Bu, cəmi 200 nümunəlik kiçik dataset ilə gözlənilən nəticədir. Validation metrikaları stabil azalma göstərir (loss 1.995-dən 1.003-ə enir), overfitting müşahidə olunmur (Δ = 0.131).

Təlim gedişatı yuxarıdakı qrafikdə vizual olaraq göstərilir. `load_best_model_at_end=True` parametri ilə ən yaxşı model avtomatik bərpa olunur. Daha böyük dataset ilə WER-in əhəmiyyətli dərəcədə yaxşılaşması gözlənilir.
```