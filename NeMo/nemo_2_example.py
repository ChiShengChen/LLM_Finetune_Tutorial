#!/usr/bin/env python3
"""
model 下載： https://catalog.ngc.nvidia.com/?filters=&orderBy=weightPopularDESC&query=&page=&pageSize=
安裝套件： pip install nemo_toolkit[all] 
或 conda install -c conda-forge nemo_toolkit
pip install nemo-run
其他諸如 pip install Cython, pip install megatron-core, pip install transformer-engine等等缺啥裝啥
# 生成數據：python nemo_2_example.py --generate-data
# 轉換模型：python nemo_2_example.py --convert-model --hf-model-path /path/to/hf_model --output-path /output/dir
# 微調模型：python nemo_2_example.py --finetune --model-path /path/to/model
# 與模型聊天：python nemo_2_example.py --interactive --model-path /path/to/model
# 評估模型：python nemo_2_example.py --evaluate --model-path /path/to/model --test-file test_data.jsonl
# 運行完整工作流程：python nemo_2_example.py --workflow
"""
"""
NeMo 不支援直接 fine-tune LLaMA3
NeMo 官方支援的 GPT 模型是 Megatron-GPT 架構，而非 Hugging Face 版本的 LLaMA3 模型。
所以這份教學依據 NeMo 官方建議使用以最合適的模型加載和生成方法
"""

import os
import torch
import json
import random
import numpy as np
from tqdm import tqdm
import time
import argparse
import pkg_resources
import sys
import tempfile

# Import the function from generate_data.py
from generate_data import generate_conversation_data

# 檢測 NeMo 版本並提供相容性
try:
    NEMO_VERSION = pkg_resources.get_distribution("nemo-toolkit").version
    NEMO_MAJOR_VERSION = int(NEMO_VERSION.split('.')[0])
    print(f"NeMo version {NEMO_VERSION} detected")
except:
    print("Warning: Could not determine NeMo version. Please ensure NeMo is properly installed.")
    NEMO_MAJOR_VERSION = 0

# 嘗試導入必要的依賴
try:
    # Common imports
    from omegaconf import OmegaConf
    
    # Version-specific imports for NeMo and PyTorch Lightning
    if NEMO_MAJOR_VERSION >= 2:
        # NeMo 2.x imports
        from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
        from lightning.pytorch.trainer import Trainer as LightningTrainer  # Use new namespace
        try:
            import megatron.core
        except ImportError:
            print("Warning: megatron.core not found. Please install: pip install megatron-core")
        try:
            import nemo_run
        except ImportError:
            print("Warning: nemo_run not found. Please install: pip install nemo-run")
    else:
        # NeMo 1.x imports
        from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
        from nemo.collections.nlp.models.text_generation import TextGenerationModel
        from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
        from nemo.utils.exp_manager import exp_manager
        from pytorch_lightning.trainer.trainer import Trainer as PyTorchLightningTrainer # Keep old for 1.x

    import pytorch_lightning as pl # Can still be used for pl.Trainer if needed for other parts or older logic

except ImportError as e:
    print(f"Error: {e}")
    print("Please make sure NeMo and its dependencies are properly installed.")
    print("Installation instructions: pip install nemo_toolkit[all]")

# 其他必要的導入
try:
    from pytorch_lightning.trainer.trainer import Trainer
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError as e:
    print(f"Warning: {e}")
    print("Some functionality may be limited.")

class LLaMA3ChatbotFineTuner:
    def __init__(self, config_path=None):
        """
        Initialize the LLaMA 3 chatbot fine-tuner.
        
        Args:
            config_path (str, optional): Path to the configuration file. If None, default config will be used.
        """
        self.model = None
        self.trainer = None
        
        # Load config or use default
        if config_path and os.path.exists(config_path):
            self.config = OmegaConf.load(config_path)
        else:
            self.config = self._get_default_config()
            
    def _get_default_config(self):
        """Create default configuration for fine-tuning."""
        # NeMo 2.x 使用不同的配置結構
        if NEMO_MAJOR_VERSION >= 2:
            config = OmegaConf.create({
                "trainer": {
                    "devices": 1,
                    "num_nodes": 1,
                    "precision": 16,
                    "accelerator": "gpu",
                    "max_epochs": 3,
                    "max_steps": -1,
                    "accumulate_grad_batches": 1,
                    "gradient_clip_val": 1.0,
                    "log_every_n_steps": 10,
                    "val_check_interval": 0.05,
                    "limit_val_batches": 50,
                },
                "exp_manager": {
                    "explicit_log_dir": "nemo_experiments",
                    "exp_dir": "nemo_experiments",
                    "name": "llama3_finetuned_chatbot",
                    "create_wandb_logger": False,
                    "wandb_logger_kwargs": {
                        "project": "llama3_chatbot",
                        "name": "llama3_finetuned"
                    },
                    "create_checkpoint_callback": False
                },
                "model": {
                    "restore_from_path": None,  # Path to pretrained LLaMA 3 checkpoint
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "global_batch_size": 8,
                    "micro_batch_size": 4,
                    "optim": {
                        "name": "distributed_fused_adam",
                        "lr": 5e-6,
                        "weight_decay": 0.1,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                    },
                    "data": {
                        "train_ds": {
                            "file_path": "path/to/train_data.jsonl",
                            "label_key": "answer",
                            "input_key": "query",
                            "num_samples": -1,
                        },
                        "validation_ds": {
                            "file_path": "path/to/val_data.jsonl",
                            "label_key": "answer",
                            "input_key": "query",
                            "num_samples": 100,
                        },
                    },
                    "chat_template": {
                        "system_prompt": "You are a helpful, harmless, and honest AI assistant.",
                        "user_prefix": "<|user|>\n",
                        "assistant_prefix": "<|assistant|>\n",
                        "end_token": "</s>",
                    }
                }
            })
        else:
            # NeMo 1.x 使用原始配置
            config = OmegaConf.create({
                "trainer": {
                    "devices": 1,
                    "num_nodes": 1,
                    "precision": 16,
                    "accelerator": "gpu",
                    "max_epochs": 3,
                    "max_steps": -1,
                    "accumulate_grad_batches": 1,
                    "gradient_clip_val": 1.0,
                    "log_every_n_steps": 10,
                    "val_check_interval": 0.05,
                    "limit_val_batches": 50,
                },
                "exp_manager": {
                    "explicit_log_dir": "nemo_experiments",
                    "exp_dir": "nemo_experiments",
                    "name": "llama3_finetuned_chatbot",
                    "create_wandb_logger": False,
                    "wandb_logger_kwargs": {
                        "project": "llama3_chatbot",
                        "name": "llama3_finetuned"
                    },
                    "create_checkpoint_callback": False
                },
                "model": {
                    "restore_from_path": None,  # Path to pretrained LLaMA 3 checkpoint
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "micro_batch_size": 4,
                    "global_batch_size": 8,
                    "learning_rate": 5e-6,
                    "weight_decay": 0.1,
                    "optim": {
                        "name": "adamw",
                        "lr": 5e-6,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 0.1,
                    },
                    "data": {
                        "train_ds": {
                            "file_path": "path/to/train_data.jsonl",
                            "label_key": "answer",
                            "input_key": "query",
                            "num_samples": -1,
                        },
                        "validation_ds": {
                            "file_path": "path/to/val_data.jsonl",
                            "label_key": "answer",
                            "input_key": "query",
                            "num_samples": 100,
                        },
                    },
                    "chat_template": {
                        "system_prompt": "You are a helpful, harmless, and honest AI assistant.",
                        "user_prefix": "<|user|>\n",
                        "assistant_prefix": "<|assistant|>\n",
                        "end_token": "</s>",
                    },
                    "create_checkpoint_callback": False
                }
            })
        return config

    def prepare_llama3_model(self, pretrained_model_path=None):
        """
        Prepare the LLaMA 3 model for fine-tuning.
        
        Args:
            pretrained_model_path (str, optional): Path to pretrained LLaMA 3 model checkpoint.
        """
        if pretrained_model_path:
            self.config.model.restore_from_path = pretrained_model_path # This sets it on the script's config

        # Check the script's config for the path to the .nemo file
        current_restore_path = self.config.model.get('restore_from_path', None)
        if not current_restore_path:
            raise ValueError("restore_from_path must be set in self.config.model to load a pretrained model.")
            
        if self.trainer is None:
            # This might happen if called outside the standard workflow. 
            # For the workflow, setup_training() should be called first.
            print("Warning: Trainer not initialized prior to prepare_llama3_model. Attempting to setup trainer now.")
            self.setup_training() # Attempt to set it up if not done.
            if self.trainer is None: # Check again
                 raise ValueError("Trainer could not be initialized. Cannot proceed with model preparation.")

        # Prepare overrides from the script's config for finetuning.
        # These will be merged by restore_from with the config loaded from the .nemo file.
        # Architectural parameters (num_layers, hidden_size, etc.) will come from the .nemo file.
        cfg_overrides_dict = {}
        script_model_cfg = self.config.model

        # Finetuning-specific parameters
        if script_model_cfg.get('optim') is not None:
            cfg_overrides_dict['optim'] = OmegaConf.to_container(script_model_cfg.optim, resolve=True)
        if script_model_cfg.get('data') is not None:
            cfg_overrides_dict['data'] = OmegaConf.to_container(script_model_cfg.data, resolve=True)
        if script_model_cfg.get('global_batch_size') is not None:
            cfg_overrides_dict['global_batch_size'] = script_model_cfg.global_batch_size
        if script_model_cfg.get('micro_batch_size') is not None:
            cfg_overrides_dict['micro_batch_size'] = script_model_cfg.micro_batch_size
        if script_model_cfg.get('chat_template') is not None:
            cfg_overrides_dict['chat_template'] = OmegaConf.to_container(script_model_cfg.chat_template, resolve=True)

        # Parallelism config from the script (use with caution, ensure compatibility with loaded model)
        # These will override the .nemo file's settings if provided.
        if script_model_cfg.get('tensor_model_parallel_size') is not None:
             cfg_overrides_dict['tensor_model_parallel_size'] = script_model_cfg.tensor_model_parallel_size
        if script_model_cfg.get('pipeline_model_parallel_size') is not None:
             cfg_overrides_dict['pipeline_model_parallel_size'] = script_model_cfg.pipeline_model_parallel_size
        
        # Ensure that the restore_from_path itself is not part of the overrides,
        # as it's the primary argument to restore_from.
        if 'restore_from_path' in cfg_overrides_dict:
            del cfg_overrides_dict['restore_from_path']

        print(f"Restoring model from {current_restore_path} with finetuning overrides for keys: {list(cfg_overrides_dict.keys())}")
        
        # Convert the overrides dictionary to an OmegaConf object,
        # ensuring it's structured with a 'model' key for proper merging.
        override_config_for_file = OmegaConf.create({'model': cfg_overrides_dict})

        # Save override_config_for_file to a temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_yaml:
            OmegaConf.save(config=override_config_for_file, f=tmp_yaml.name)
            override_config_file_path = tmp_yaml.name

        try:
            self.model = MegatronGPTModel.restore_from(
                restore_path=current_restore_path,
                trainer=self.trainer,
                override_config_path=override_config_file_path # Pass path to YAML file
            )
        finally:
            # Clean up the temporary file
            if override_config_file_path and os.path.exists(override_config_file_path):
                os.remove(override_config_file_path)
        
        # After restoration, self.model.cfg is the config from .nemo merged with overrides.
        # The script's self.config.model might be less complete or slightly out of sync.
        # For finetuning, the model will use its own self.model.cfg.
        # If the script needs to refer to the definitive model config, it should use self.model.cfg.
        # For example, you might want to update the script's config for consistency if other parts rely on it:
        # self.config.model = self.model.cfg # This merges the loaded and overridden config back to the script's main config.

        return self.model
    
    def setup_training(self):
        """Set up the PyTorch Lightning trainer for fine-tuning."""
        if NEMO_MAJOR_VERSION >= 2:
            # NeMo 2.x 使用新的 Trainer 設置
            trainer_config = self.config.trainer.copy()
            trainer_config["logger"] = False # Disable default logger

            self.trainer = LightningTrainer( # Use the aliased new Trainer
                **trainer_config
            )
            
            # NeMo 2.x 中 exp_manager 可能有變化
            try:
                from nemo.utils.exp_manager import exp_manager
                exp_manager(self.trainer, self.config.exp_manager)
            except ImportError:
                print("Warning: exp_manager not found in NeMo 2.x. Please check NeMo documentation.")
                
        else:
            # NeMo 1.x 使用原始設置
            strategy = NLPDDPStrategy(
                no_ddp_communication_hook=True,
                find_unused_parameters=False,
            )
            
            self.trainer = PyTorchLightningTrainer( # Use the aliased old Trainer for NeMo 1.x
                strategy=strategy,
                **self.config.trainer
            )
            
            exp_manager(self.trainer, self.config.exp_manager)
        
        return self.trainer
    
    def finetune(self):
        """Start fine-tuning the model."""
        if self.model is None:
            raise ValueError("Model not initialized. Call prepare_llama3_model first.")
        
        if self.trainer is None:
            self.setup_training()
            
        self.trainer.fit(self.model)
        
    def save_model(self, output_dir):
        """
        Save the fine-tuned model.
        
        Args:
            output_dir (str): Directory to save the model to.
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "llama3_finetuned.nemo")
        self.model.save_to(model_path)
        
        return model_path
    
    def load_model(self, model_path):
        """
        Load a fine-tuned model.
        
        Args:
            model_path (str): Path to the saved model file.
        """
        if NEMO_MAJOR_VERSION >= 2:
            # NeMo 2.x 使用 MegatronGPTModel 加載
            try:
                # 確保 AppState 已初始化
                from nemo.utils.app_state import AppState
                app_state = AppState()
                if app_state.model_parallel_size is None:
                    app_state.model_parallel_size = 1
                
                self.model = MegatronGPTModel.restore_from(model_path)
            except Exception as e:
                print(f"Error loading model with MegatronGPTModel: {e}")
                print("Trying alternative loading method...")
                
                try:
                    from omegaconf import OmegaConf
                    cfg = OmegaConf.create({"restore_from_path": model_path})
                    self.model = MegatronGPTModel(cfg=cfg)
                except Exception as e2:
                    print(f"Alternative loading also failed: {e2}")
                    raise ValueError(f"Failed to load model from {model_path}")
        else:
            # NeMo 1.x 使用 TextGenerationModel
            try:
                self.model = TextGenerationModel.restore_from(model_path)
            except Exception as e:
                print(f"Error loading model with TextGenerationModel: {e}")
                print("Trying to load with MegatronGPTModel...")
                
                try:
                    self.model = MegatronGPTModel.restore_from(model_path)
                except Exception as e2:
                    print(f"Alternative loading also failed: {e2}")
                    raise ValueError(f"Failed to load model from {model_path}")
        
        return self.model
    
    def chat(self, user_input, system_prompt=None, max_length=100, 
             temperature=0.8, top_k=50, top_p=0.95, 
             repetition_penalty=1.2, min_length=0, 
             beam_width=1, length_penalty=1.0, 
             random_seed=None, use_greedy=False):
        """
        Generate a response using the fine-tuned model.
        
        Args:
            user_input (str): User's input text.
            system_prompt (str, optional): Custom system prompt. If None, uses the default from config.
            max_length (int): Maximum length of the generated response.
            temperature (float): Controls randomness in generation. Higher values (> 1.0) increase randomness,
                               lower values (< 1.0) make output more deterministic. Set to 0 for greedy decoding.
            top_k (int): Keeps only the top k tokens with highest probability (top-k filtering).
                        Set to 0 to disable this filter.
            top_p (float): Keeps the top tokens whose cumulative probability exceeds top_p (nucleus sampling).
                         Set to 1.0 to disable this filter.
            repetition_penalty (float): Penalizes repetition. Values > 1.0 discourage repetition.
            min_length (int): Minimum length of the generated response.
            beam_width (int): Beam size for beam search. Set to 1 for greedy or sampling strategies.
            length_penalty (float): Penalizes sequences based on their length. Values < 1.0 favor shorter sequences.
            random_seed (int, optional): Seed for random number generation to make output deterministic.
            use_greedy (bool): If True, uses greedy decoding instead of sampling (ignores temperature).
            
        Returns:
            str: Generated response.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Load or train a model first.")
            
        system = system_prompt if system_prompt else self.config.model.chat_template.system_prompt
        user_prefix = self.config.model.chat_template.user_prefix
        assistant_prefix = self.config.model.chat_template.assistant_prefix
        
        # Format input according to LLaMA 3 chat template
        prompt = f"{system}\n{user_prefix}{user_input}\n{assistant_prefix}"
        
        # 根據 NeMo 版本使用不同的生成方法
        if NEMO_MAJOR_VERSION >= 2:
            # NeMo 2.x 使用更新的生成 API
            try:
                # 使用 NeMo 2.x 新的生成參數格式
                response = self.model.generate(
                    inputs=[prompt],
                    length_params={"max_length": max_length, "min_length": min_length},
                    sampling_params={
                        "use_greedy": use_greedy or temperature == 0,
                        "temperature": 0.0 if use_greedy or temperature == 0 else temperature,
                        "top_k": 1 if use_greedy or temperature == 0 else top_k,
                        "top_p": 1.0 if use_greedy or temperature == 0 else top_p,
                        "repetition_penalty": repetition_penalty,
                        "length_penalty": length_penalty,
                        "beam_width": beam_width if not (use_greedy or temperature == 0) else 1
                    }
                )
                
                # 處理生成的文本
                generated_text = response[0]
                
            except (AttributeError, TypeError) as e:
                print(f"Warning: New generation API failed with error: {e}")
                print("Trying fallback generation method...")
                
                # 嘗試使用備用方法
                try:
                    response = self.model.generate_samples(
                        [prompt],
                        max_length=max_length,
                        min_length=min_length,
                        temperature=0.0 if use_greedy or temperature == 0 else temperature,
                        top_k=1 if use_greedy or temperature == 0 else top_k,
                        top_p=1.0 if use_greedy or temperature == 0 else top_p,
                        repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty,
                        beam_width=beam_width if not (use_greedy or temperature == 0) else 1,
                        add_BOS=False
                    )
                    generated_text = response[0]
                except Exception as e2:
                    print(f"Fallback generation also failed: {e2}")
                    print("Using model forward pass as last resort...")
                    
                    # 最後嘗試直接使用模型前向傳遞
                    inputs = self.model.tokenizer.text_to_ids(prompt)
                    input_tensor = torch.tensor([inputs], device=self.model.device)
                    
                    with torch.no_grad():
                        output_tensor = self.model(input_tensor)
                        generated_ids = output_tensor[0].argmax(dim=-1)[-max_length:]
                    
                    generated_text = self.model.tokenizer.ids_to_text(generated_ids.tolist())
        else:
            # NeMo 1.x 使用 TextGenerationModel
            gen_params = {
                'text': [prompt],
                'max_length': max_length,
                'min_length': min_length
            }
            
            # 處理貪婪解碼和採樣參數
            if use_greedy or temperature == 0:
                gen_params.update({
                    'temperature': 0.0,
                    'top_k': 1,
                    'top_p': 1.0,
                    'add_BOS': False,
                    'all_probs': False,
                    'compute_logprobs': False,
                })
            else:
                gen_params.update({
                    'temperature': temperature,
                    'top_k': top_k,
                    'top_p': top_p,
                    'repetition_penalty': repetition_penalty,
                    'beam_width': beam_width,
                    'length_penalty': length_penalty,
                    'add_BOS': False,
                    'all_probs': False,
                    'compute_logprobs': False,
                })
                
                if random_seed is not None:
                    gen_params['random_seed'] = random_seed
            
            # 使用 generate 方法
            try:
                response = self.model.generate(**gen_params)
                generated_text = response[0]
            except Exception as e:
                print(f"Error in generate method: {e}")
                print("Trying alternative method...")
                
                # 嘗試其他生成方法
                try:
                    response = self.model.predict_step(gen_params)
                    generated_text = response[0]
                except Exception as e2:
                    print(f"Alternative generation failed: {e2}")
                    raise ValueError("Failed to generate text with this model.")
        
        # 處理生成的文本
        # 提取助理的回應
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
        
        # 移除終止令牌
        end_token = self.config.model.chat_template.end_token
        if end_token in generated_text:
            generated_text = generated_text.split(end_token)[0]
            
        return generated_text.strip()

    def evaluate_model(self, test_file, metrics=None, num_examples=None):
        """
        Evaluate the model on test data using various metrics.
        
        Args:
            test_file (str): Path to test data file in JSONL format.
            metrics (list, optional): List of metrics to compute. If None, uses default metrics.
            num_examples (int, optional): Number of examples to evaluate. If None, evaluates all.
            
        Returns:
            dict: Dictionary of evaluation results.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Load or train a model first.")
            
        # Default metrics if none provided
        if metrics is None:
            metrics = ["bleu", "exact_match", "response_length", "latency"]
            
        # Try to download NLTK resources if needed
        try:
            nltk.download('punkt', quiet=True)
        except:
            print("Could not download NLTK resources. BLEU score may not be available.")
            
        # Load test data
        test_data = []
        with open(test_file, 'r') as f:
            for line in f:
                test_data.append(json.loads(line))
                
        if num_examples is not None:
            test_data = test_data[:num_examples]
            
        results = {metric: [] for metric in metrics}
        smoothie = SmoothingFunction().method1  # For BLEU calculation
        
        print(f"Evaluating model on {len(test_data)} examples...")
        for example in tqdm(test_data):
            query = example.get("query", "")
            ground_truth = example.get("answer", "")
            
            # Measure generation latency
            start_time = time.time()
            try:
                generated = self.chat(query)
                end_time = time.time()
                latency = end_time - start_time
                
                # Calculate metrics
                if "bleu" in metrics:
                    reference = [ground_truth.split()]
                    candidate = generated.split()
                    try:
                        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
                        results["bleu"].append(bleu_score)
                    except:
                        # If BLEU calculation fails, use 0
                        results["bleu"].append(0)
                        
                if "exact_match" in metrics:
                    exact_match = 1.0 if generated.strip() == ground_truth.strip() else 0.0
                    results["exact_match"].append(exact_match)
                    
                if "response_length" in metrics:
                    results["response_length"].append(len(generated.split()))
                    
                if "latency" in metrics:
                    results["latency"].append(latency)
            except Exception as e:
                print(f"Error generating response for query: {query}")
                print(f"Error: {e}")
                # 如果生成失敗，添加默認值
                for metric in metrics:
                    results[metric].append(0)
        
        # Compute average for each metric
        evaluation_results = {}
        for metric in results:
            if results[metric]:  # 確保列表不為空
                evaluation_results[metric] = np.mean(results[metric])
            else:
                evaluation_results[metric] = 0.0
            
        # Print results
        print("\nEvaluation Results:")
        for metric, value in evaluation_results.items():
            print(f"{metric}: {value:.4f}")
            
        return evaluation_results
    
    def interactive_chat(self, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2):
        """
        Start an interactive chat session with the model.
        
        Args:
            temperature (float): Controls randomness in generation (default: 0.7).
            top_k (int): Keeps only the top k tokens with highest probability (default: 50).
            top_p (float): Nucleus sampling threshold (default: 0.9).
            repetition_penalty (float): Penalizes repetition (default: 1.2).
        """
        if self.model is None:
            raise ValueError("Model not loaded. Load or train a model first.")
            
        print(f"Starting interactive chat with LLaMA 3 (using NeMo {NEMO_MAJOR_VERSION}.x)")
        print("=" * 50)
        
        # Allow user to customize system prompt
        system_prompt = input("Custom system prompt (press Enter for default): ")
        if not system_prompt:
            system_prompt = self.config.model.chat_template.system_prompt
            print(f"Using default: \"{system_prompt}\"")
            
        # Allow user to customize generation parameters
        print("\nGeneration settings:")
        print(f"- Temperature: {temperature} (higher = more random)")
        print(f"- Top-k: {top_k} (0 to disable)")
        print(f"- Top-p: {top_p} (1.0 to disable)")
        print(f"- Repetition penalty: {repetition_penalty} (higher = less repetition)")
        change_settings = input("\nChange generation settings? [y/N]: ").lower()
        
        if change_settings == 'y':
            try:
                temp_input = input(f"Temperature ({temperature}): ")
                if temp_input.strip():
                    temperature = float(temp_input)
                
                top_k_input = input(f"Top-k ({top_k}): ")
                if top_k_input.strip():
                    top_k = int(top_k_input)
                
                top_p_input = input(f"Top-p ({top_p}): ")
                if top_p_input.strip():
                    top_p = float(top_p_input)
                
                rep_input = input(f"Repetition penalty ({repetition_penalty}): ")
                if rep_input.strip():
                    repetition_penalty = float(rep_input)
                    
                print("\nUpdated generation settings:")
                print(f"- Temperature: {temperature}")
                print(f"- Top-k: {top_k}")
                print(f"- Top-p: {top_p}")
                print(f"- Repetition penalty: {repetition_penalty}")
            except ValueError as e:
                print(f"Error parsing values: {e}. Using defaults.")
                temperature = 0.7
                top_k = 50
                top_p = 0.9
                repetition_penalty = 1.2
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("Ending chat session.")
                    break
                    
                # Add to conversation history
                conversation_history.append({"role": "user", "content": user_input})
                
                # Format multi-turn conversations if needed
                if len(conversation_history) > 1:
                    context = "\n".join([
                        f"{self.config.model.chat_template.user_prefix}{msg['content']}" 
                        if msg['role'] == 'user' else 
                        f"{self.config.model.chat_template.assistant_prefix}{msg['content']}"
                        for msg in conversation_history[:-1]
                    ])
                    context += f"\n{self.config.model.chat_template.user_prefix}{user_input}"
                    response = self.chat(
                        context, 
                        system_prompt=system_prompt,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty
                    )
                else:
                    response = self.chat(
                        user_input, 
                        system_prompt=system_prompt,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty
                    )
                    
                # Add to conversation history
                conversation_history.append({"role": "assistant", "content": response})
                
                print(f"\nAssistant: {response}")
            except KeyboardInterrupt:
                print("\nInterrupted by user. Ending chat session.")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again with different input or settings.")

def convert_hf_llama_to_nemo(hf_model_path, output_path):
    """
    Convert a HuggingFace LLaMA model to NeMo format.
    
    Args:
        hf_model_path (str): Path to HuggingFace model.
        output_path (str): Path to save the NeMo model.
    """
    try:
        # 根據 NeMo 版本使用不同的轉換工具
        if NEMO_MAJOR_VERSION >= 2:
            # NeMo 2.x 使用新的轉換模組
            try:
                from nemo.collections.nlp.models.language_modeling.megatron.convert_hf_checkpoint import convert_hf_checkpoint
                
                print(f"Converting HuggingFace model {hf_model_path} to NeMo format (using NeMo 2.x API)...")
                convert_hf_checkpoint(
                    hf_model_path,
                    output_path,
                    model_class="gpt",
                    hf_model_name="llama"
                )
                print(f"Model converted and saved to {output_path}")
            except ImportError:
                print("Error: Could not import convert_hf_checkpoint from NeMo 2.x.")
                print("Please check if you have the latest NeMo version with conversion support.")
                print("For NeMo 2.x, the import path might have changed.")
        else:
            # NeMo 1.x 使用舊的轉換模組
            try:
                from nemo.collections.nlp.modules.common.megatron.utils import convert_hf_checkpoint
                
                print(f"Converting HuggingFace model {hf_model_path} to NeMo format (using NeMo 1.x API)...")
                convert_hf_checkpoint(
                    hf_model_path,
                    output_path,
                    model_class="gpt",
                    hf_model_name="llama"
                )
                print(f"Model converted and saved to {output_path}")
            except ImportError:
                print("Error: Could not import convert_hf_checkpoint from NeMo 1.x.")
                print("Please check if you have the correct NeMo version.")
    except Exception as e:
        print(f"Error converting model: {str(e)}")
        print("Make sure you have the latest NeMo toolkit with HuggingFace conversion support.")
        print("You may need to install additional dependencies for model conversion.")

def complete_workflow_example():
    print("DEBUG: Entered complete_workflow_example()")
    """
    Complete workflow example for fine-tuning LLaMA 3 with NeMo.
    """
    print("===== Complete LLaMA 3 Fine-tuning Workflow =====")
    print(f"Using NeMo version: {NEMO_VERSION if 'NEMO_VERSION' in globals() else 'Unknown'}")
    print("1. Generating training and validation data...")
    
    # Generate data
    training_data = "llama3_training_data.jsonl"
    validation_data = "llama3_validation_data.jsonl"
    if not os.path.exists(training_data) or not os.path.exists(validation_data):
        generate_conversation_data(150, training_data)
        generate_conversation_data(50, validation_data)
    
    # Initialize fine-tuner
    finetuner = LLaMA3ChatbotFineTuner()
    
    # Update data paths
    finetuner.config.model.data.train_ds.file_path = training_data
    finetuner.config.model.data.validation_ds.file_path = validation_data
    
    print("\n2. Model and training configuration:")
    print("Training epochs:", finetuner.config.trainer.max_epochs)
    
    if NEMO_MAJOR_VERSION >= 2:
        # NeMo 2.x 配置
        print("Learning rate:", finetuner.config.model.optim.lr)
    else:
        # NeMo 1.x 配置
        print("Learning rate:", finetuner.config.model.learning_rate)
        
    print("Batch size:", finetuner.config.model.global_batch_size)
    
    # Model preparation
    llama_path = "/home/aidan/Time_series_benchmark/LLM_tutorial/megatron_gpt_345m.nemo"  # Hardcoded path
    if llama_path:
        # Call setup_training() before prepare_llama3_model()
        print("\n3. Setting up training environment (trainer)...")
        finetuner.setup_training()

        # DEBUG: Inspect the override config structure
        print("\nDEBUG: Constructing and inspecting the effective override configuration that would be used:")
        # Replicate the logic from prepare_llama3_model to build cfg_overrides_dict
        cfg_overrides_dict_debug = {}
        script_model_cfg_debug = finetuner.config.model
        if script_model_cfg_debug.get('optim') is not None:
            cfg_overrides_dict_debug['optim'] = OmegaConf.to_container(script_model_cfg_debug.optim, resolve=True)
        if script_model_cfg_debug.get('data') is not None:
            cfg_overrides_dict_debug['data'] = OmegaConf.to_container(script_model_cfg_debug.data, resolve=True)
        if script_model_cfg_debug.get('global_batch_size') is not None:
            cfg_overrides_dict_debug['global_batch_size'] = script_model_cfg_debug.global_batch_size
        if script_model_cfg_debug.get('micro_batch_size') is not None:
            cfg_overrides_dict_debug['micro_batch_size'] = script_model_cfg_debug.micro_batch_size
        if script_model_cfg_debug.get('chat_template') is not None:
            cfg_overrides_dict_debug['chat_template'] = OmegaConf.to_container(script_model_cfg_debug.chat_template, resolve=True)
        if script_model_cfg_debug.get('tensor_model_parallel_size') is not None:
             cfg_overrides_dict_debug['tensor_model_parallel_size'] = script_model_cfg_debug.tensor_model_parallel_size
        if script_model_cfg_debug.get('pipeline_model_parallel_size') is not None:
             cfg_overrides_dict_debug['pipeline_model_parallel_size'] = script_model_cfg_debug.pipeline_model_parallel_size
        if 'restore_from_path' in cfg_overrides_dict_debug:
            del cfg_overrides_dict_debug['restore_from_path']
        
        override_config_for_file_debug = OmegaConf.create({'model': cfg_overrides_dict_debug})
        print(OmegaConf.to_yaml(override_config_for_file_debug))
        # END DEBUG

        print(f"\n4. Preparing model from {llama_path} using the trainer...")
        finetuner.prepare_llama3_model(llama_path)
        
        confirm = input("\nStart fine-tuning now? [y/N]: ")
        if confirm.lower() == 'y':
            print("\n5. Starting fine-tuning...")
            finetuner.finetune()
            
            print("\n6. Saving fine-tuned model...")
            model_path = finetuner.save_model("fine_tuned_model")
            print(f"Model saved to {model_path}")
            
            print("\n7. Evaluating model...")
            finetuner.evaluate_model(validation_data)
        else:
            print("Fine-tuning skipped.")
    else:
        print("\nNo model path provided. To fine-tune, you need:")
        print("1. A pretrained LLaMA 3 model in NeMo format")
        print("2. If you have a HuggingFace model, convert it using convert_hf_llama_to_nemo()")
        
    print("\nComplete workflow finished!")
    # print("DEBUG: Inside complete_workflow_example(), returned from complete_workflow_example()") # Removed previous debug

def main():
    print("DEBUG: Entered main() function") # New very first line debug
    """Run LLaMA 3 chatbot fine-tuning framework."""
    parser = argparse.ArgumentParser(description="LLaMA 3 Chatbot Fine-tuning Framework")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic conversation data")
    parser.add_argument("--data-samples", type=int, default=150, help="Number of conversation examples to generate")
    parser.add_argument("--convert-model", action="store_true", help="Convert HuggingFace model to NeMo format")
    parser.add_argument("--hf-model-path", type=str, help="Path to HuggingFace model")
    parser.add_argument("--output-path", type=str, help="Path to save the converted model")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune a model")
    parser.add_argument("--model-path", type=str, help="Path to pretrained model")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--test-file", type=str, help="Path to test data file")
    parser.add_argument("--workflow", action="store_true", help="Run complete workflow example")
    parser.add_argument("--check-nemo", action="store_true", help="Check NeMo installation and version")
    
    args = parser.parse_args()
    
    # 檢查 NeMo 安裝和版本
    if args.check_nemo or not any([args.generate_data, args.convert_model, args.finetune, 
                                    args.interactive, args.evaluate, args.workflow]):
        print("\n=== NeMo Version and Installation Check ===")
        try:
            # 檢測 NeMo 版本
            nemo_version = pkg_resources.get_distribution("nemo-toolkit").version
            major_version = int(nemo_version.split('.')[0])
            print(f"NeMo version: {nemo_version} (Major version: {major_version})")
            
            # 檢查關鍵依賴
            dependencies = [
                ("pytorch_lightning", "PyTorch Lightning"),
                ("torch", "PyTorch"),
                ("omegaconf", "OmegaConf"),
                ("megatron.core", "Megatron Core"),
                ("nemo_run", "NeMo Run")
            ]
            
            print("\nDependency check:")
            for module_name, display_name in dependencies:
                try:
                    module = __import__(module_name.split('.')[0])
                    if '.' in module_name:
                        try:
                            for part in module_name.split('.')[1:]:
                                module = getattr(module, part)
                            print(f"✓ {display_name} - Installed")
                        except AttributeError:
                            print(f"✗ {display_name} - Missing or incorrect version")
                    else:
                        module_version = getattr(module, "__version__", "Unknown")
                        print(f"✓ {display_name} - Installed (Version: {module_version})")
                except (ImportError, ModuleNotFoundError):
                    print(f"✗ {display_name} - Not installed")
            
            print("\nInstallation appears to be:")
            if major_version >= 2:
                if 'megatron.core' in sys.modules and 'nemo_run' in sys.modules:
                    print("GOOD - NeMo 2.x with required dependencies")
                else:
                    print("PARTIAL - NeMo 2.x but missing some dependencies (megatron.core or nemo_run)")
            else:
                print("GOOD - NeMo 1.x")
                
            print("\nFor NeMo 2.x, ensure you have installed:")
            print("pip install megatron-core nemo-run")
            
        except (ImportError, pkg_resources.DistributionNotFound):
            print("NeMo (nemo-toolkit) is not installed!")
            print("Please install NeMo using: pip install nemo_toolkit[all]")
            
        if args.check_nemo:
            return
    
    # 檢查參數衝突和依賴關係
    if args.finetune and not args.model_path:
        parser.error("--finetune requires --model-path to be specified")
        
    if args.convert_model and (not args.hf_model_path or not args.output_path):
        parser.error("--convert-model requires both --hf-model-path and --output-path to be specified")
        
    if args.evaluate and not args.test_file:
        parser.error("--evaluate requires --test-file to be specified")
        
    if args.interactive and not args.model_path and not args.finetune:
        parser.error("--interactive requires --model-path to be specified when not used with --finetune")
    
    if args.workflow:
        print("DEBUG: Inside main(), about to call complete_workflow_example()")
        complete_workflow_example()
        print("DEBUG: Inside main(), returned from complete_workflow_example()")
        return
    
    if args.generate_data:
        print("Generating synthetic conversation data...")
        training_data = generate_conversation_data(args.data_samples, "llama3_training_data.jsonl")
        validation_data = generate_conversation_data(args.data_samples // 3, "llama3_validation_data.jsonl")
        return
        
    if args.convert_model:
        convert_hf_llama_to_nemo(args.hf_model_path, args.output_path)
        return
    
    # Initialize the fine-tuner
    finetuner = LLaMA3ChatbotFineTuner()
    
    if args.finetune:
        # 這裡不需要再檢查model_path了，因為在參數解析階段已經檢查過
        finetuner.prepare_llama3_model(args.model_path)
        finetuner.setup_training()
        finetuner.finetune()
        finetuner.save_model("fine_tuned_model")
    
    if args.interactive:
        if args.model_path:
            finetuner.load_model(args.model_path)
        elif not finetuner.model:
            # 如果同時使用了--finetune，這裡不會執行到，因為已經載入模型
            print("Using fine-tuned model for chat")
        
        finetuner.interactive_chat()
    
    if args.evaluate:
        # 這裡不需要再檢查test_file了，因為在參數解析階段已經檢查過
        if args.model_path:
            finetuner.load_model(args.model_path)
        elif not finetuner.model:
            parser.error("--evaluate requires a loaded model. Use --model-path to specify it.")
            
        finetuner.evaluate_model(args.test_file)
    
    # If no specific action was chosen, run the data generation example
    if not any([args.generate_data, args.convert_model, args.finetune, 
                args.interactive, args.evaluate, args.workflow, args.check_nemo]):
        print("No action specified. Generating synthetic conversation data as an example.")
        
        # Generate training and validation data
        print("Generating synthetic conversation data for training...")
        training_conversations = generate_conversation_data(150, "llama3_training_data.jsonl")
        validation_conversations = generate_conversation_data(50, "llama3_validation_data.jsonl")
        
        # Print some examples
        print("\nExample training conversations:")
        for i, conv in enumerate(training_conversations[:3]):
            print(f"\nConversation {i+1}:")
            print(f"User: {conv['user_message']}")
            print(f"Assistant: {conv['assistant_message']}")
        
        print("\nTraining data generated successfully!")
        print(f"Training data: {len(training_conversations)} conversations saved to llama3_training_data.jsonl")
        print(f"Validation data: {len(validation_conversations)} conversations saved to llama3_validation_data.jsonl")
        print("\nNext steps:")
        print("1. Download a pretrained LLaMA 3 model")
        print("2. Convert it to NeMo format if needed")
        print("3. Fine-tune the model: python nemo.py --finetune --model-path /path/to/model")
        print("4. Chat with your model: python nemo.py --interactive --model-path fine_tuned_model/llama3_finetuned.nemo")
        print("\nTo check your NeMo installation: python nemo.py --check-nemo") 

if __name__ == "__main__":
    print("DEBUG: __name__ is __main__, about to call main()") # New debug before calling main
    main() 