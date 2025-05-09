import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# 1. Load a pre-trained model
model_name = "gpt2"  # You can use other GPT-2 variants like "gpt2-medium", "gpt2-large", "gpt2-xl"
max_seq_length = 1024 # Choose any valid length
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use token if using gated models like meta-llama/Llama-2-7b-hf
)

# 2. Add LoRA adapters for PEFT
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0. Suggested 8, 16, 32, 64, 128
    target_modules = None,  # Modules to apply LoRA to. For GPT2.
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# 3. Prepare the dataset
# We'll use the "imdb" dataset as an example for text generation.
# You can replace this with your own dataset.
dataset_name = "imdb"
dataset = load_dataset(dataset_name, split="train[:1%]") # Using a small subset for demonstration

# Preprocess the dataset
def preprocess_function(examples):
    # For GPT-2, we typically concatenate text and let the model learn.
    # We'll just use the 'text' field from the imdb dataset.
    # Ensure your dataset has a 'text' field or adapt this function.
    return tokenizer(examples["text"], truncation=True, max_length=max_seq_length, padding="max_length")

processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4, # Number of processes for mapping
    remove_columns=dataset.column_names
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 4. Fine-tune the model
# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-unsloth",
    num_train_epochs=1,  # For demonstration, use a small number of epochs
    per_device_train_batch_size=2, # Reduce if OOM
    gradient_accumulation_steps=4, # Increase to save memory
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=50,
    fp16=not torch.cuda.is_bf16_supported(), # Use fp16 if bf16 is not available
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit" if load_in_4bit else "adamw_torch", # Use 8-bit AdamW optimizer if 4-bit quantization is used
    lr_scheduler_type="linear",
    seed=3407,
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=data_collator,
)

# Start training
print("Starting training...")
trainer.train()
print("Training finished.")

# 5. Save the model
output_model_path = "./gpt2_finetuned_unsloth_model"
print(f"Saving model to {output_model_path}")
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)
print("Model saved.")

# To load the saved model for inference:
# from unsloth import FastLanguageModel
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = output_model_path, # YOUR FINETUNED MODEL
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
# )
# FastLanguageModel.for_inference(model) # Enable native 2x faster inference
#
# You can then use the model for generation, for example:
# from transformers import pipeline
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
# result = pipe("Once upon a time,", max_length=50)
# print(result)

print("\nExample script for fine-tuning GPT-2 with Unsloth is complete.")
print("Make sure you have the necessary packages installed: pip install torch unsloth datasets transformers[torch]")
