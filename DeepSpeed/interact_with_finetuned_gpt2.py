from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def interact_with_model():
    # --- Configuration ---
    # Adjust this path if your fine-tuned model is saved elsewhere
    model_path = "./my_gpt2_finetuned_model_hf_format"
    
    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Fine-tuned Model and Tokenizer ---
    print(f"Loading tokenizer from: {model_path}")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Please ensure the tokenizer was saved correctly in the specified path.")
        return

    print(f"Loading model from: {model_path}")
    try:
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model.to(device) # Move model to the selected device
        model.eval()     # Set the model to evaluation mode
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model was saved correctly in the specified path.")
        return

    print("\\nModel and tokenizer loaded successfully.\\n")

    # --- Interact with the Model ---
    while True:
        prompt = input("Enter your prompt (or type 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break

        if not prompt.strip():
            print("Prompt cannot be empty.")
            continue

        try:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

            # Generate text
            # You can experiment with these generation parameters
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=150,          # Maximum length of the generated text
                temperature=0.7,       # Controls randomness. Lower is more deterministic.
                top_k=50,              # Considers only the top k tokens for sampling.
                top_p=0.95,            # Nucleus sampling: considers tokens with cumulative probability >= top_p.
                num_return_sequences=1, # Number of different sequences to generate
                pad_token_id=tokenizer.eos_token_id # Set pad_token_id to eos_token_id for open-ended generation
            )

            # Decode and print the generated text
            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                print(f"--- Generated Sequence {generated_sequence_idx+1} ---")
                text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
                # To display only the newly generated part (excluding the prompt):
                # text_only_generated = text[len(prompt):]
                # print(text_only_generated)
                print(text)
                print("-----------------------------------\\n")

        except Exception as e:
            print(f"Error during text generation: {e}")

if __name__ == "__main__":
    interact_with_model() 