# Unsloth 大型語言模型微調範例

本專案包含使用 [Unsloth](https://github.com/unslothai/unsloth) 函式庫微調大型語言模型（LLM）的範例腳本。Unsloth 能夠顯著加速 LLM 的訓練並減少記憶體使用。

## 目錄
- [Unsloth 大型語言模型微調範例](#unsloth-大型語言模型微調範例)
  - [目錄](#目錄)
  - [專案結構](#專案結構)
  - [環境設置](#環境設置)
    - [1. Python 環境](#1-python-環境)
    - [2. 安裝依賴套件](#2-安裝依賴套件)
  - [腳本說明與使用](#腳本說明與使用)
    - [1. GPT-2 微調 (`gpt2_unsloth_finetune.py`)](#1-gpt-2-微調-gpt2_unsloth_finetunepy)
    - [2. LLaMA 3 微調 (`llama3_unsloth_finetune.py`)](#2-llama-3-微調-llama3_unsloth_finetunepy)
    - [3. LLaMA 3 推論 (`llama3_inference.py`)](#3-llama-3-推論-llama3_inferencepy)
  - [注意事項](#注意事項)

## 專案結構
```
LLM_unsloth_tutorial/
├── gpt2_unsloth_finetune.py     # GPT-2 微調腳本
├── llama3_unsloth_finetune.py   # LLaMA 3 微調腳本
├── llama3_inference.py          # LLaMA 3 推論腳本
└── README.md                    # 本說明文件
```

## 環境設置

### 1. Python 環境
建議使用 Python 3.9 或更高版本。推薦使用 Conda 創建獨立的虛擬環境：
```bash
conda create -n unsloth_env python=3.10
conda activate unsloth_env
```

### 2. 安裝依賴套件
在您的虛擬環境中，使用 pip 安裝必要的函式庫。

首先，安裝 PyTorch。請參考 [PyTorch 官方網站](https://pytorch.org/get-started/locally/) 以取得適合您系統（CUDA 版本等）的安裝指令。例如：
```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

然後，安裝 Unsloth 和其他相關套件：
```bash
pip install "unsloth[cu121-ampere-torch240]" # 請根據您的 CUDA 版本和 GPU 架構選擇，例如 cu118, cu121, ampere, Hopper 等
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git" # 可選擇安裝最新版
pip install datasets transformers accelerate bitsandbytes sentencepiece
# xformers (可選，但推薦用於進一步優化)
pip install xformers
```
**注意**: `unsloth` 的安裝指令可能會因您的 CUDA 版本、PyTorch 版本和 GPU 架構而異。請參考 [Unsloth GitHub](https://github.com/unslothai/unsloth) 以獲取最新的安裝指南。

## 腳本說明與使用

### 1. GPT-2 微調 (`gpt2_unsloth_finetune.py`)
此腳本使用 Unsloth 微調標準的 GPT-2 模型。

**功能**:
- 從 Hugging Face Hub 載入預訓練的 GPT-2 模型。
- 使用 LoRA (Low-Rank Adaptation) 技術進行參數高效微調。
- 使用 `imdb` 資料集進行示範性文本生成微調（您可以替換為自己的資料集）。
- 儲存微調後的模型和 tokenizer。

**如何運行**:
```bash
python LLM_unsloth_tutorial/gpt2_unsloth_finetune.py
```

**主要可調參數** (在腳本內修改):
- `model_name`: 要微調的 GPT-2 變體 (例如 `"gpt2"`, `"gpt2-medium"`)。
- `dataset_name`: 用於訓練的 Hugging Face 資料集名稱。
- `max_seq_length`: 輸入序列的最大長度。
- `load_in_4bit`: 是否使用 4-bit 量化 (預設為 `False`，因為 GPT-2 的原始 Unsloth 範例可能不總是啟用它)。
- `TrainingArguments`: 包含訓練週期數 (`num_train_epochs`)、批次大小 (`per_device_train_batch_size`) 等。

**輸出**:
- 微調後的模型和 tokenizer 會儲存在腳本中 `output_model_path` 指定的路徑 (預設為 `./gpt2_finetuned_unsloth_model`)。

### 2. LLaMA 3 微調 (`llama3_unsloth_finetune.py`)
此腳本使用 Unsloth 微調 LLaMA 3 8B 模型，並採用 4-bit 量化以節省記憶體。

**功能**:
- 載入 `unsloth/llama-3-8b-bnb-4bit` 模型 (已預先進行 4-bit 量化)。
- 應用 LoRA 適配器。
- 使用 `imdb` 資料集進行示範。
- 儲存微調後的模型和 tokenizer。

**如何運行**:
```bash
python LLM_unsloth_tutorial/llama3_unsloth_finetune.py
```

**主要可調參數**:
- `model_name`: 預設為 `"unsloth/llama-3-8b-bnb-4bit"`。
- `max_seq_length`: 輸入序列的最大長度。
- `TrainingArguments`: 訓練相關參數。

**輸出**:
- 微調後的模型和 tokenizer 會儲存在 `./llama3-finetuned` 資料夾。

### 3. LLaMA 3 推論 (`llama3_inference.py`)
此腳本使用先前由 `llama3_unsloth_finetune.py` 微調並儲存的 LLaMA 3 模型進行文本生成。

**功能**:
- 載入微調後的 LLaMA 3 模型。
- 啟用 Unsloth 的快速推論模式。
- 使用 Hugging Face `pipeline` 進行文本生成。

**運行前提**:
- 必須先成功運行 `llama3_unsloth_finetune.py` 並產生 `./llama3-finetuned` 模型文件。

**如何運行**:
```bash
python LLM_unsloth_tutorial/llama3_inference.py
```

**主要可調參數**:
- `model_path`: 指向微調模型的路徑 (預設為 `"./llama3-finetuned"`)。
- `prompt`: 用於生成的初始文本提示。
- `pipe()` 中的參數: 例如 `max_new_tokens`, `temperature`, `top_p` 等，用於控制生成文本的風格和長度。

**輸出**:
- 在終端打印生成的文本。

## 注意事項
- **GPU 需求**: 微調大型語言模型通常需要具有足夠 VRAM 的 NVIDIA GPU。4-bit 量化和 Unsloth 有助於降低需求，但對於 LLaMA 3 8B 這類模型，仍建議使用至少 16GB VRAM 的 GPU。GPT-2 的需求較低。
- **Unsloth 編譯快取**: Unsloth 會在首次運行時編譯優化核心。有時快取可能導致問題。如果遇到與 Unsloth 編譯相關的錯誤，可以嘗試刪除快取目錄：
  ```bash
  rm -rf ./LLM_unsloth_tutorial/unsloth_compiled_cache /tmp/unsloth_compiled_cache ~/.cache/unsloth
  ```
  然後重新運行腳本。
- **客製化**: 這些腳本是範例，您可以根據自己的需求修改資料集、模型參數和訓練配置。
- **Hugging Face Token**: 如果您使用的模型是私有的或需要授權 (gated models)，您可能需要在 `FastLanguageModel.from_pretrained` 中提供 Hugging Face token (例如 `token = "hf_..."`)。
 