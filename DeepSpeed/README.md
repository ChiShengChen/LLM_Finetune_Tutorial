# DeepSpeed GPT-2 Fine-tuning 教學

本專案提供了一個使用 DeepSpeed 對 GPT-2 模型進行 Fine-tuning 並與之互動的範例。

## 專案結構

```
LLM_deepspeed_tutorial/
├── gpt2_deepspeed_finetune.py       # Fine-tuning 腳本
├── interact_with_finetuned_gpt2.py  # 與 Fine-tuned 模型互動的腳本
└── my_gpt2_finetuned_model_hf_format/ # Fine-tuning 後模型儲存的預設資料夾 (執行完 fine-tuning 後生成)
```

## 環境準備

1.  **Python 環境**: 建議使用 Conda 建立獨立的 Python 環境。
    ```bash
    conda create -n deepspeed_env python=3.10 -y
    conda activate deepspeed_env
    ```
2.  **硬體**: 需要 NVIDIA GPU 並已安裝相應的 CUDA 驅動程式。

## 安裝步驟

1.  **安裝 PyTorch**:
    根據您的 CUDA 版本從 PyTorch 官方網站 ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)) 安裝 PyTorch。例如：
    ```bash
    # 請根據您的 CUDA 版本選擇合適的指令
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

2.  **安裝 DeepSpeed 和 Transformers**:
    ```bash
    pip install deepspeed transformers
    ```

3.  **安裝 MPI (用於 DeepSpeed 通訊後端)**:
    建議使用 Conda 安裝，以確保環境一致性：
    ```bash
    conda install -c conda-forge openmpi mpi4py -y
    ```
    如果選擇系統級 MPI (例如 `sudo apt install libopenmpi-dev`)，請確保其與 `mpi4py` 編譯相容。

4.  **安裝 CUDA 編譯工具 (用於 DeepSpeed JIT 編譯自訂 CUDA 核心)**:
    DeepSpeed 在首次執行時可能會即時編譯 (JIT compile) 一些優化的 CUDA 核心 (如 FusedAdam)。這需要完整的 CUDA 開發工具鏈。如果遇到缺少 `.h` 檔案 (如 `cusparse.h`, `cublas_v2.h`, `cusolverDn.h`) 或編譯器問題，請透過 Conda 安裝：
    ```bash
    conda install -c conda-forge cxx-compiler cuda-nvcc libcusparse-dev libcublas-dev libcusolver-dev -y
    ```
    安裝完畢後，建議清除 PyTorch 的擴充編譯快取，以確保下次執行時重新編譯：
    ```bash
    rm -rf ~/.cache/torch_extensions/
    ```
    (快取路徑可能因 PyTorch 版本和 CUDA 版本略有不同，請根據錯誤日誌中的路徑調整)

## 如何進行模型 Fine-tuning

1.  **準備資料集**:
    *   目前的 `gpt2_deepspeed_finetune.py` 腳本中使用 `get_dummy_dataset` 函數生成虛擬資料。
    *   您需要修改此函數，或替換它來載入和預處理您自己的真實資料集。資料集應格式化為模型可以理解的輸入。

2.  **設定 Fine-tuning 參數**:
    *   **基礎模型**: 在 `gpt2_deepspeed_finetune.py` 中，您可以修改 `model_name = 'gpt2'` 來選擇不同的 GPT-2 變體 (如 `gpt2-medium`, `gpt2-large`) 或其他 Hugging Face 上的預訓練模型。
    *   **DeepSpeed 設定**: `config_params` 字典包含了 DeepSpeed 的設定，例如：
        *   `train_batch_size`: 訓練的批次大小。
        *   `gradient_accumulation_steps`: 梯度累積步數。
        *   `optimizer`: 優化器類型和參數 (如學習率 `lr`)。
        *   `scheduler`: 學習率排程。
        *   `fp16`: 是否啟用混合精度訓練 (通常建議啟用以加速並節省記憶體)。
        *   `zero_optimization`: ZeRO 優化設定，特別是 `stage` (0, 1, 2, 3) 的選擇，會影響記憶體優化程度和訓練方式。
    *   **模型儲存路徑**: fine-tuned 完的模型和 tokenizer 預設會儲存在 `gpt2_deepspeed_finetune.py` 中 `save_directory` 變數指定的路徑 (預設為 `./my_gpt2_finetuned_model_hf_format`)。

3.  **執行 Fine-tuning**:
    在 `LLM_deepspeed_tutorial` 目錄下執行：
    ```bash
    deepspeed gpt2_deepspeed_finetune.py
    ```
    *   如果您的 DeepSpeed 設定是寫在一個 JSON 檔案中 (例如 `ds_config.json`)，則執行指令應為：
        `deepspeed gpt2_deepspeed_finetune.py --deepspeed_config ds_config.json`
    *   訓練過程中，您會看到每個 epoch 的 loss 變化。
    *   訓練完成後，fine-tuned 的模型和 tokenizer 會儲存到您指定的 `save_directory`。

## 如何與 Fine-tuned 模型互動

1.  **確認模型路徑**:
    *   `interact_with_finetuned_gpt2.py` 腳本預期 fine-tuned 模型儲存在 `model_path = "./my_gpt2_finetuned_model_hf_format"`。
    *   如果您的模型儲存在不同路徑，請修改此腳本中的 `model_path` 變數。

2.  **執行互動腳本**:
    在 `LLM_deepspeed_tutorial` 目錄下執行：
    ```bash
    python interact_with_finetuned_gpt2.py
    ```
    腳本會載入模型和 tokenizer，然後提示您輸入文字。輸入後，模型會生成接續的文本。輸入 `quit` 可退出程式。

## 注意事項

*   **資源需求**: Fine-tuning LLM 對 GPU 記憶體和計算資源有較高要求。請根據您的硬體調整 `train_batch_size` 和 DeepSpeed ZeRO stage 等設定。
*   **錯誤排除**:
    *   如果遇到 `ModuleNotFoundError`，請確保已安裝所有必要的 Python 套件。
    *   如果遇到 CUDA 核心編譯錯誤 (如缺少 `.h` 檔)，請確認已按照上述步驟安裝 `cxx-compiler`, `cuda-nvcc` 及相關的 `-dev` 套件 (如 `libcusparse-dev` 等)，並清除了 PyTorch 擴充快取。
    *   MPI 相關錯誤通常與 `mpi4py` 或底層 MPI 函式庫 (Open MPI / MPICH) 的安裝有關。

希望這份說明能幫助您順利執行此專案！ 