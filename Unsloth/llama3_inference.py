import torch
from unsloth import FastLanguageModel
from transformers import pipeline

# 路徑為你剛剛訓練後儲存的模型資料夾
model_path = "./llama3-finetuned"
max_seq_length = 2048
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# 1. 載入模型與 tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,  # 若使用了 4-bit 微調
)

# 2. 啟用推論模式（會加速運算）
FastLanguageModel.for_inference(model)

# 3. 使用 HuggingFace pipeline 產生文字
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

# 4. 測試生成範例
prompt = "Once upon a time, in a world of dragons and wizards,"
output = pipe(
    prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.2,
    eos_token_id=tokenizer.eos_token_id,
)

# 5. 輸出結果
print("\n📜 Generated Text:")
print(output[0]["generated_text"])
