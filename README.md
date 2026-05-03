
```markdown
# 🎙️ Azerbaijani ASR — Whisper Fine-Tuning with LoRA

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📌 Layihə Haqqında

### 🎯 Məqsəd
Bu layihə Azərbaycan dili üçün **Automatic Speech Recognition (ASR)** sisteminin qurulmasını hədəfləyir.  
Baza model olaraq `LocalDoc/azerbaijani-whisper-small` istifadə olunur və **LoRA adapterləri** ilə Azərbaycan dilinə uyğunlaşdırılır.

---

## ✨ Əsas Xüsusiyyətlər

- 🔧 LoRA fine-tuning — <1% parametr ilə adaptasiya
- 📊 Real-time monitoring — WER / CER izlənməsi
- 🛡️ Overfitting monitorinqi — avtomatik best checkpoint seçimi
- ⚡ Optimallaşdırılmış təlim — FP16 + gradient accumulation
- 🗣️ Azərbaycan dili fonetik dəstəyi (`ə, ö, ü, ğ, ç, ş`)

---

## 🧠 Texnologiya Stackı

| Komponent | Texnologiya |
|----------|-------------|
| Training | PyTorch 2.0+, Transformers, PEFT (LoRA) |
| Audio | librosa, soundfile, Whisper feature extractor |
| Metrics | WER/CER (jiwer / evaluate) |
| Vizualizasiya | Matplotlib |
| Infrastruktur | Google Colab T4 GPU |

---

## 📊 Dataset
🗣️ İstifadə olunan Dataset

Bu layihədə istifadə olunan audio dataset:

Common Voice Scripted Speech 25.0 — Azerbaijani

📌 Platform: Mozilla Common Voice (Mozilla Data Collective)
🧾 Dataset type: Scripted Speech (ASR üçün read speech)
🗣️ Dil: Azerbaijani
📦 License: CC0-1.0 (public domain)
🎯 Task: Automatic Speech Recognition (ASR)
🔗 Dataset Link

👉 https://mozilladatacollective.com/datasets/cmn29hqvk015ko107fblsr5ay


## 🏗️ Model Arxitekturası

### 📦 Baza Model
```

LocalDoc/azerbaijani-whisper-small
├── Encoder: 12 transformer layers
├── Decoder: 12 transformer layers
├── Hidden size: 768
├── Parametrlər: ~244M
└── Pretrained: Multilingual (100+ dil)

```
--- 
### ⚙️ LoRA Konfiqurasiyası

| Parametr | Dəyər | İzah |
|----------|------|------|
| r | 16 | balanslı rank |
| lora_alpha | 32 | scaling factor |
| lora_dropout | 0.05 | overfitting qarşısı |
| target_modules | q_proj, v_proj | attention tuning |
| trainable params | ~1.2M (0.5%) | effektiv adaptasiya |

---

### 💡 Niyə LoRA?

| Yanaşma | Params | VRAM | Vaxt | Risk |
|----------|--------|------|------|------|
| Full FT | 244M | 25GB+ | 2–3h | Yüksək |
| LoRA | 1.2M | 6GB | 20min | Aşağı |
| Frozen | 60M | 12GB | 1h | Orta |

---

## 📊 Nəticələr

### 🧪 Test Performansı

| Model | WER | CER | Dataset |
|------|-----|-----|--------|
| Baza | `base_wer` | `base_cer` | 0 AZ |
| Fine-tuned | `ft_wer` | `ft_cer` | 200 AZ |

> 📈 Improvement: `improvement`%

---

### 📉 Training Metrics

| Metrik | Start | Best | Step |
|--------|------|------|------|
| WER | `start_wer` | `best_wer` | `best_step` |
| CER | `start_cer` | `best_cer` | `best_step` |
| Train Loss | 2.849 → 0.872 | ✔ | 200 |
| Val Loss | 1.995 → 1.003 | ✔ | 200 |

---

### 🧯 Overfitting Analizi

```

Train Loss: 2.849 → 0.872  ↓69%
Val Loss:   1.995 → 1.003  ↓50%

Δ (gap): 0.131
Overfitting: ❌ YOX

````

---

## 📈 Vizualizasiya

### 📉 Loss Qrafiki
### 📊 Model Performansı (Test Set)

![!\[Validation Metrics\](results/validation_wer_cer.png)](results/training_progress.png)

### 📊 WER / CER
![!\[Validation Metrics\](results/validation_wer_cer.png)](results/training_progress.png)
| Model             | WER (%) ↓ | CER (%) ↓ |
|------------------|----------|----------|
| Baza Model       | 22.65    | 7.00     |
| Fine-tuned Model | 23.06    | 7.65     |

> ⚠️ Qeyd: Daha aşağı dəyərlər daha yaxşı performansı göstərir.
---

## 🧪 Nümunə Nəticələr

### 🟢 Ən Yaxşı

| # | WER | Cümlə |
|--|-----|-------|
| 1 | 0.000 | bunun səbəbləri müxtəlifdir |
| 2 | 0.000 | mahmud ailənin yeganə oğul övladı olmuşdur |

---

### 🔴 Ən Pis

| # | WER | Ref → Pred |
|--|-----|------------|
| 1 | 0.800 | qədim alban tarixi → bəli bəli qədim... |
| 2 | 0.778 | komanda yoldaşları → omanda... |

---

## ⚙️ Quraşdırma

### 📦 Tələblər
- Python 3.10+
- CUDA 11.8+
- 8–15GB VRAM (T4 tövsiyə olunur)

---

### 1️⃣ Repo

```bash
git clone https://github.com/<username>/azerbaijani-asr-whisper.git
cd azerbaijani-asr-whisper
````

---

### 2️⃣ Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux
venv\Scripts\activate     # Windows
```

---

### 3️⃣ Install

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Dataset Format

```
clips/
train.tsv
dev.tsv
test.tsv
```

TSV:

```
path	sentence
audio_001.wav	bunun səbəbləri müxtəlifdir
```

---

### 5️⃣ Training

```bash
python fine_tune.py
```

---

### 6️⃣ Evaluation

```bash
python evaluate.py
```

---

## 📁 Project Structure

```
azerbaijani-asr-whisper/
├── fine_tune.py
├── evaluate.py
├── whisper_finetune_az.ipynb
├── results/
│   ├── training_loss.png
│   ├── validation_wer_cer.png
│   └── b3_model_comparison.csv
└── whisper-small-az-lora/
```

---

## 🔬 Analiz

### ✅ Güclü tərəflər

* Sadə cümlələrdə çox yüksək accuracy
* Stabil training (no overfitting)
* Efficient compute usage

### ❌ Zəif tərəflər

* Uzun cümlələrdə WER yüksək
* OOV (named entities)
* Noise sensitivity

---

## 🚀 Production Roadmap

* ONNX / TensorRT optimization
* Streaming inference (VAD-based)
* Data expansion (10K+ hours)
* Larger Whisper model (Medium/Large)

---

## ⚠️ Limitasiyalar

* Kiçik dataset (200 samples)
* Whisper Small model limitləri
* Noise augmentation yoxdur

---

## 📄 License

MIT License

---

## 👤 Contact

GitHub: [Rafi Veyisov](https://github.com/rafiveyisov)

---
