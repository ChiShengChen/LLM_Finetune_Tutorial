import torch
import deepspeed
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader, TensorDataset

# 1. Configuration (Replace with your actual ds_config.json or parameters)
config_params = {
    "train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-5,
            "warmup_num_steps": 100
        }
    },
    "fp16": {
        "enabled": True # Or False, if not using mixed precision
    },
    "zero_optimization": {
        "stage": 0 # Or 1, 2, 3 depending on your ZeRO optimization level
    }
}

# --- Dummy Data Preparation (Replace with your actual data loading and preprocessing) ---
def get_dummy_dataset(tokenizer, num_samples=100, max_length=50):
    dummy_texts = ["This is a sample sentence." for _ in range(num_samples)]
    # Tokenize all texts
    inputs = tokenizer(dummy_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    # Create labels (for language modeling, labels are typically the input_ids shifted)
    labels = inputs.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100 # Ignore padding tokens in loss calculation
    return TensorDataset(inputs.input_ids, inputs.attention_mask, labels)

# 2. Load Model and Tokenizer
model_name = 'gpt2' # You can choose other GPT-2 variants like 'gpt2-medium'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# Add a padding token if it doesn't exist (GPT-2 usually doesn't have one by default)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer)) # Resize embeddings if new tokens were added

# --- Prepare Dummy Dataset ---
dummy_dataset = get_dummy_dataset(tokenizer)
# Use a portion for training for this example
train_dataloader = DataLoader(dummy_dataset, batch_size=config_params["train_batch_size"])

# 3. Initialize DeepSpeed Engine
# Make sure deepspeed is installed and ds_config.json is configured or use config_params
model_engine, optimizer, _, scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params=config_params # or use `config="path/to/ds_config.json"`
)

print(f"Using DeepSpeed ZeRO Stage: {model_engine.zero_optimization_stage()}")
print(f"FP16 enabled: {model_engine.fp16_enabled()}")

# 4. Training Loop
num_epochs = 3 # Example number of epochs
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model_engine.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(model_engine.local_rank)
        attention_mask = attention_mask.to(model_engine.local_rank)
        labels = labels.to(model_engine.local_rank)

        outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        model_engine.backward(loss)
        model_engine.step() # Includes optimizer.step() and scheduler.step()

        total_loss += loss.item()
        if step % 10 == 0: # Print loss every 10 steps
            print(f"  Step {step}/{len(train_dataloader)}, Loss: {loss.item()}")

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"  Average Training Loss: {avg_train_loss}")

    # --- Evaluation (Optional, add your evaluation logic here) ---
    # model_engine.eval()
    # ...

# 5. Saving Checkpoints (DeepSpeed handles this internally based on config, but you can also save manually)
# model_engine.save_checkpoint("my_model_checkpoints") # This creates a directory

# To save the final model for Hugging Face Transformers compatibility:
save_directory = "./my_gpt2_finetuned_model_hf_format" # Define the save directory

if hasattr(model_engine, 'zero_optimization_stage') and model_engine.zero_optimization_stage() == 3:
    # If using ZeRO Stage 3, need to consolidate weights first on rank 0
    if hasattr(model_engine, 'global_rank') and model_engine.global_rank == 0:
        # model_engine.module is the original Hugging Face model
        model_engine.module.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print(f"Model and tokenizer saved to {save_directory} (ZeRO Stage 3 consolidated from rank 0)")
    elif not hasattr(model_engine, 'global_rank'): # Fallback if global_rank somehow not set with ZeRO-3
        # This case should ideally not happen with DeepSpeed ZeRO-3 setup
        model_engine.module.save_pretrained(save_directory) 
        tokenizer.save_pretrained(save_directory)
        print(f"Model and tokenizer saved to {save_directory} (ZeRO Stage 3, global_rank not found - potential issue)")
elif hasattr(model_engine, 'global_rank') and model_engine.global_rank == 0:
    # For ZeRO stages 0, 1, 2 or no ZeRO, save from rank 0
    model_engine.module.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved by rank 0 to {save_directory} (ZeRO Stage < 3 or no ZeRO)")
elif not hasattr(model_engine, 'global_rank'):
    # Fallback for truly non-distributed single-process execution not managed by deepspeed launcher (less common for DeepSpeed)
    # Or if DeepSpeed is not used at all (though the script is a DeepSpeed script)
    model_engine.module.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved to {save_directory} (Likely non-distributed or DeepSpeed not fully initialized for distributed info)")


print(f"Training complete. Fine-tuned model and tokenizer attempted to be saved. Check logs above for status and path: {save_directory}.")

# To run this script:
# deepspeed gpt2_deepspeed_finetune.py --deepspeed_config ds_config.json (if using a file)
# Or if config_params is embedded:
# deepspeed gpt2_deepspeed_finetune.py 