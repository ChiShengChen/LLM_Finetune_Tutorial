import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Step 1: Load LLaMA 3-8B with 4-bit quantization
model_name = "unsloth/llama-3-8b-bnb-4bit"
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,  # Auto detect: bf16 / fp16
    load_in_4bit=True,
)

# Step 2: Apply LoRA adapters (optional, but recommended)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # For LLaMA 3
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
)

# Step 3: Load and preprocess dataset
dataset = load_dataset("imdb", split="train[:1%]")  # small subset for testing

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )

processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=4,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 4: Define training arguments
training_args = TrainingArguments(
    output_dir="./llama3-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    fp16=True,  # assume fp16 support
    lr_scheduler_type="linear",
    seed=3407,
)

# Step 5: Initialize Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=data_collator,
)

print("🚀 Starting fine-tuning with LLaMA 3...")
trainer.train()
print("✅ Training finished.")

# Step 6: Save the fine-tuned model
output_dir = "./llama3-finetuned"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"📦 Model saved to {output_dir}")
