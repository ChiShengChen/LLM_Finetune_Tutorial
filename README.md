# LLM Fine-tuning and Application Tutorial

This project provides comprehensive examples and scripts for fine-tuning and interacting with Large Language Models (LLMs) using three popular toolkits: **DeepSpeed**, **NVIDIA NeMo**, and **Unsloth**. Each subfolder contains scripts, sample data, and detailed instructions for setting up your environment, running fine-tuning, and performing inference.

---

## Project Structure

```
LLM_Finetune_Tutorial-main/
├── DeepSpeed/    # Fine-tuning and inference with DeepSpeed (GPT-2)
├── NeMo/         # Fine-tuning and application with NVIDIA NeMo (LLaMA, GPT, etc.)
├── Unsloth/      # Fast fine-tuning and inference with Unsloth (GPT-2, LLaMA 3)
```

---

## 1. DeepSpeed

Fine-tune GPT-2 models efficiently using DeepSpeed, with support for ZeRO optimization and mixed precision.

### Key Files
- `gpt2_deepspeed_finetune.py`: Fine-tuning script for GPT-2.
- `interact_with_finetuned_gpt2.py`: Script to interact with your fine-tuned model.

### Setup

1. **Create a Conda environment:**
   ```bash
   conda create -n deepspeed_env python=3.10 -y
   conda activate deepspeed_env
   ```

2. **Install dependencies:**
   - Install PyTorch (choose the correct CUDA version from [PyTorch website](https://pytorch.org/get-started/locally/)).
   - Install DeepSpeed and Transformers:
     ```bash
     pip install deepspeed transformers
     ```
   - Install MPI and CUDA development tools as needed.

3. **Dataset:**  
   The script uses a dummy dataset by default. Replace or modify the `get_dummy_dataset` function to use your own data.

4. **Run fine-tuning:**
   ```bash
   deepspeed gpt2_deepspeed_finetune.py
   ```

5. **Interact with the model:**
   ```bash
   python interact_with_finetuned_gpt2.py
   ```

**Note:** Adjust batch size, ZeRO stage, and other DeepSpeed configs according to your hardware.

---

## 2. NVIDIA NeMo

End-to-end workflow for fine-tuning and applying LLMs (e.g., LLaMA) using NVIDIA NeMo.

### Key Files
- `nemo_2_example.py`: Main script for model conversion, fine-tuning, evaluation, and interactive chat.
- `generate_data.py`: Script to generate synthetic training/validation data.
- `llama3_training_data.jsonl`, `llama3_validation_data.jsonl`: Example datasets.

### Setup

1. **Create a Conda environment:**
   ```bash
   conda create -n nemo python=3.10
   conda activate nemo
   ```

2. **Install NeMo and dependencies:**
   ```bash
   pip install nemo_toolkit[all]
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install Cython megatron-core transformer-engine pytorch-lightning omegaconf tqdm nltk
   ```

3. **Check installation:**
   ```bash
   python nemo_2_example.py --check-nemo
   ```

### Workflow

- **Generate data:**  
  `python generate_data.py`  
  or  
  `python nemo_2_example.py --generate-data --data-samples 200`

- **Model conversion (if needed):**  
  Convert Hugging Face models to NeMo format:
  ```bash
  python nemo_2_example.py --convert-model --hf-model-path /path/to/hf_model --output-path /path/to/model.nemo
  ```

- **Fine-tune:**  
  ```bash
  python nemo_2_example.py --finetune --model-path /path/to/pretrained.nemo
  ```

- **Interactive chat:**  
  ```bash
  python nemo_2_example.py --interactive --model-path /path/to/fine_tuned_model.nemo
  ```

- **Evaluate:**  
  ```bash
  python nemo_2_example.py --evaluate --model-path /path/to/fine_tuned_model.nemo --test-file /path/to/test_data.jsonl
  ```

**Note:** Ensure CUDA, PyTorch, and NeMo versions are compatible. Some dependencies (Apex, Transformer Engine) may require building from source.

---

## 3. Unsloth

Fast and memory-efficient fine-tuning of LLMs (GPT-2, LLaMA 3) using Unsloth, with support for 4-bit quantization and LoRA adapters.

### Key Files
- `gpt2_unsloth_finetune.py`: Fine-tune GPT-2 with Unsloth.
- `llama3_unsloth_finetune.py`: Fine-tune LLaMA 3 (8B, 4-bit) with Unsloth.
- `llama3_inference.py`: Inference script for fine-tuned LLaMA 3.

### Setup

1. **Create a Conda environment:**
   ```bash
   conda create -n unsloth_env python=3.10
   conda activate unsloth_env
   ```

2. **Install PyTorch** (choose the correct CUDA version):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install Unsloth and dependencies:**
   ```bash
   pip install "unsloth[cu121-ampere-torch240]"
   pip install datasets transformers accelerate bitsandbytes sentencepiece xformers
   ```

### Usage

- **Fine-tune GPT-2:**
  ```bash
  python gpt2_unsloth_finetune.py
  ```

- **Fine-tune LLaMA 3:**
  ```bash
  python llama3_unsloth_finetune.py
  ```

- **Inference with LLaMA 3:**
  ```bash
  python llama3_inference.py
  ```

**Note:**  
- For LLaMA 3 8B, at least a 16GB VRAM GPU is recommended.
- If you encounter Unsloth compilation cache issues, clear the cache directories as described in the Unsloth README.
- You may need a Hugging Face token for gated models.

---

## Troubleshooting

- **CUDA/Driver Issues:** Ensure all CUDA, driver, and library versions are compatible.
- **Missing Dependencies:** Install all required Python packages and system libraries as indicated in each subfolder's README.
- **Model Paths:** Adjust script variables to point to your actual model and data locations.
- **Resource Limits:** Adjust batch sizes and quantization settings to fit your hardware.

---

## References

- [DeepSpeed](https://www.deepspeed.ai/)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

---

This tutorial aims to help you quickly get started with LLM fine-tuning and application using state-of-the-art open-source tools. For detailed options and advanced usage, please refer to the individual READMEs and official documentation of each toolkit.