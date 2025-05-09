# LLM 大型語言模型微調與應用教程 (基於 NeMo)

本教程提供了一個使用 NVIDIA NeMo 工具套件來微調和應用大型語言模型（LLM）的完整流程，特別是以 LLaMA 類模型為例。

## 目錄結構

```
LLM_tutorial/
├── nemo_experiments/      # NeMo 實驗結果和日誌的默認輸出目錄
├── apex/                  # (可選) NVIDIA Apex 庫源碼（如果從源碼編譯）
├── TransformerEngine/     # (可選) NVIDIA Transformer Engine 庫源碼（如果從源碼編譯）
├── __pycache__/           # Python 編譯緩存
├── generate_data.py       # 用於生成合成對話數據的腳本
├── nemo_2_example.py      # 核心腳本：包含模型轉換、微調、評估和互動聊天功能
├── megatron_gpt_345m.nemo # 範例 NeMo 格式的預訓練模型 (用戶需自行準備或轉換)
├── llama3_training_data.jsonl # 生成的範例訓練數據
├── llama3_validation_data.jsonl # 生成的範例驗證數據
└── README.md              # 本說明文件
```

## 環境設置

1.  **創建 Conda 環境** (推薦):
    ```bash
    conda create -n nemo python=3.10  # NeMo 可能對特定 Python 版本有要求，請查閱官方文件
    conda activate nemo
    ```

2.  **安裝 NeMo 工具套件**:
    強烈建議安裝 `nemo_toolkit[all]` 以包含所有依賴。
    ```bash
    pip install nemo_toolkit[all]
    ```
    或者，如果遇到問題，可以嘗試從 PyPI 單獨安裝核心包，然後根據需要補充：
    ```bash
    pip install nemo_toolkit
    ```

3.  **安裝特定版本的 PyTorch**:
    NeMo 通常依賴特定版本的 PyTorch，請參考 NeMo 官方文檔以獲取與您 NeMo 版本兼容的 PyTorch 和 CUDA 版本。例如：
    ```bash
    # 示例命令，版本號可能需要調整
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

4.  **安裝其他必要依賴**:
    腳本 `nemo_2_example.py` 會在運行初期提示可能缺少的關鍵依賴。根據提示進行安裝，例如：
    ```bash
    pip install Cython
    pip install megatron-core
    pip install transformer-engine  # 可能需要從源碼編譯以匹配您的 CUDA 版本
    pip install pytorch-lightning # NeMo 的核心依賴
    pip install omegaconf tqdm nltk
    ```
    對於 `transformer-engine` 和 `Apex` (如果 `MegatronGPTModel` 需要)，有時從源碼編譯可以更好地匹配您的環境，特別是 CUDA 版本。編譯指令通常在其各自的 GitHub 倉庫中提供。

5.  **檢查 NeMo 安裝**:
    可以使用 `nemo_2_example.py` 腳本的 `--check-nemo` 選項來檢查 NeMo 版本和關鍵依賴的安裝情況：
    ```bash
    python nemo_2_example.py --check-nemo
    ```

## 使用流程

### 1. 生成對話數據 (可選)

如果您沒有現成的對話數據集，可以使用 `generate_data.py` 來生成合成的 JSONL 格式數據。

*   **生成數據**:
    ```bash
    python generate_data.py
    ```
    這會在當前目錄下生成 `llama3_training_data.jsonl` (150個樣本) 和 `llama3_validation_data.jsonl` (50個樣本)。

*   **客製化數據生成**:
    `generate_data.py` 腳本內部可以修改生成樣本的數量和內容模板。`nemo_2_example.py` 也可以調用此功能：
    ```bash
    python nemo_2_example.py --generate-data --data-samples 200
    ```
    這會生成200個訓練樣本和相應數量的驗證樣本。

生成的數據格式如下 (JSONL，每行一個JSON對象):
```json
{"query": "用戶的提問或發言", "answer": "期望的助手回答"}
```

### 2. 準備預訓練模型

*   **獲取 `.nemo` 格式模型**:
    微調需要一個 `.nemo` 格式的預訓練模型。您可以從 NVIDIA NGC Catalog 下載官方提供的模型，或將其他格式（如 Hugging Face）的模型轉換為 `.nemo` 格式。
    本教程中提供了一個範例路徑指向 `megatron_gpt_345m.nemo`。**請將此替換為您實際的 `.nemo` 模型路徑。**

*   **(可選) Hugging Face 模型轉換為 NeMo 格式**:
    如果您的預訓練模型是 Hugging Face 格式，可以使用 `nemo_2_example.py` 中的轉換功能。
    ```bash
    python nemo_2_example.py --convert-model \
                             --hf-model-path /path/to/your/huggingface_model_directory \
                             --output-path /path/to/output/converted_model.nemo
    ```
    請確保您的 NeMo 版本支持 LLaMA 模型的轉換。

### 3. 微調模型

使用 `nemo_2_example.py` 腳本進行微調。

*   **配置微調參數**:
    主要的微調配置在 `nemo_2_example.py` 腳本的 `LLaMA3ChatbotFineTuner` 類的 `_get_default_config()` 方法中定義。您可以根據需要修改以下參數：
    *   `trainer`: 訓練器相關配置 (如 `devices`, `max_epochs`, `precision` 等)。
    *   `exp_manager`: 實驗管理配置 (如日誌目錄 `exp_dir`, 實驗名稱 `name` 等)。
    *   `model`: 模型相關配置:
        *   `restore_from_path`: **重要！** 在腳本中或通過命令行參數 `--model-path` 指定預訓練的 `.nemo` 模型路徑。
        *   `optim`: 優化器配置 (如 `lr` 學習率)。
        *   `data`: 數據集配置，包含訓練集和驗證集的 `file_path` (指向生成的 JSONL 文件)。
        *   `chat_template`: 對話格式模板。
        *   並行策略: `tensor_model_parallel_size`, `pipeline_model_parallel_size` (請謹慎設置以匹配您的硬件和模型)。

*   **啟動微調**:
    ```bash
    python nemo_2_example.py --finetune --model-path /path/to/your/pretrained.nemo
    ```
    *   `--model-path`: 指向您準備好的 `.nemo` 預訓練模型。腳本會使用 `_get_default_config()` 中定義的數據路徑 (`llama3_training_data.jsonl`, `llama3_validation_data.jsonl`)。如果您的數據文件名稱不同，請修改腳本中的默認路徑。
    *   微調完成後，模型將被保存在 `nemo_experiments/<experiment_name>/<version>/checkpoints/` 目錄下，通常文件名類似 `llama3_finetuned.nemo`。腳本也會嘗試將最後一個 checkpoint 保存到 `fine_tuned_model/llama3_finetuned.nemo`。

### 4. 與微調後的模型互動聊天

*   **啟動互動模式**:
    ```bash
    python nemo_2_example.py --interactive --model-path /path/to/your/fine_tuned_model.nemo
    ```
    *   `--model-path`: 指向您微調後保存的 `.nemo` 模型。
    *   您可以自定義系統提示語和生成參數。

### 5. 評估模型 (可選)

*   **準備評估數據**: 格式與訓練數據相同 (JSONL)。
*   **啟動評估**:
    ```bash
    python nemo_2_example.py --evaluate \
                             --model-path /path/to/your/fine_tuned_model.nemo \
                             --test-file /path/to/your/test_data.jsonl
    ```
    *   評估指標包括 BLEU 分數、精確匹配率、響應長度和延遲。

### 6. 運行完整工作流程 (教學示例)

腳本提供了一個 `--workflow` 選項，用於演示一個簡化的端到端流程：生成數據 -> 準備預訓練模型 (使用腳本中硬編碼的路徑) -> 微調 -> 保存 -> 評估。

*   **啟動完整工作流程**:
    ```bash
    # 確保 NVIDIA_PYTORCH_VERSION 環境變量已設置 (如果您的 NeMo 版本需要)
    # 例如: export NVIDIA_PYTORCH_VERSION=24.01
    python nemo_2_example.py --workflow
    ```
    **注意**: 此工作流程中的預訓練模型路徑 (`megatron_gpt_345m.nemo`) 是硬編碼的。請確保該文件存在於指定路徑，或修改腳本以指向您的模型。系統會提示您是否開始微調。

## 故障排除與注意事項

*   **CUDA 版本**: 確保您的 CUDA 版本、NVIDIA 驅動、PyTorch CUDA 版本以及 NeMo 依賴 (如 Transformer Engine, Apex) 的 CUDA 編譯版本相互兼容。這是最常見的問題來源。
*   **環境變量**: 某些 NeMo 版本或其依賴可能需要特定的環境變量，例如 `NVIDIA_PYTORCH_VERSION`。
*   **配置文件**: OmegaConf 用於管理配置。如果遇到 `ConfigAttributeError` (如 `Key '...' is not in struct`)，通常表示模型配置的結構或內容與預期不符。檢查 `.nemo` 模型內部配置和您的覆蓋配置是否正確。
*   **Apex 和 Transformer Engine**: 這些庫通常需要針對您的特定 CUDA 版本進行編譯才能獲得最佳性能和兼容性。如果 `pip install` 版本不起作用，請嘗試從源碼編譯。
*   **NeMo 版本**: NeMo API 在不同版本之間可能存在差異。本教程中的 `nemo_2_example.py` 試圖處理 NeMo 1.x 和 2.x 的一些差異，但主要針對較新版本進行測試。
