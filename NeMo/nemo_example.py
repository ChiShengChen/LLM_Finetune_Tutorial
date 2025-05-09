#!/usr/bin/env python3
"""
安裝套件： pip install nemo_toolkit[all] 
或 conda install -c conda-forge nemo_toolkit
pip install nemo-run
其他諸如   pip install Cython, pip install megatron-core等等缺啥裝啥
# 生成數據：python nemo_example.py --generate-data
# 轉換模型：python nemo_example.py --convert-model --hf-model-path /path/to/hf_model --output-path /output/dir
# 微調模型：python nemo_example.py --finetune --model-path /path/to/model
# 與模型聊天：python nemo_example.py --interactive --model-path /path/to/model
# 評估模型：python nemo_example.py --evaluate --model-path /path/to/model --test-file test_data.jsonl
# 運行完整工作流程：python nemo_example.py --workflow
"""
"""
NeMo 不支援直接 fine-tune LLaMA3
NeMo 官方支援的 GPT 模型是 Megatron-GPT 架構，而非 Hugging Face 版本的 LLaMA3 模型。
所以這份教學依據 NeMo 官方建議使用 TextGenerationModel 替代 MegatronGPTModel
"""

import os
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.text_generation import TextGenerationModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils.exp_manager import exp_manager
from pytorch_lightning.trainer.trainer import Trainer
import json
import random
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from tqdm import tqdm
import time
import argparse

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
                }
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
            self.config.model.restore_from_path = pretrained_model_path
            
        # Initialize the model - use MegatronGPTModel for training
        self.model = MegatronGPTModel(cfg=self.config.model)
        
        return self.model
    
    def setup_training(self):
        """Set up the PyTorch Lightning trainer for fine-tuning."""
        strategy = NLPDDPStrategy(
            no_ddp_communication_hook=True,
            find_unused_parameters=False,
        )
        
        self.trainer = Trainer(
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
        # Use TextGenerationModel instead of MegatronGPTModel for inference
        self.model = TextGenerationModel.restore_from(model_path)
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
        
        # Set up generation parameters
        gen_params = {
            'text': [prompt],
            'max_length': max_length
        }
        
        # Handle special case for greedy decoding
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
            # Add sampling parameters 
            gen_params.update({
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'repetition_penalty': repetition_penalty,
                'min_length': min_length,
                'beam_width': beam_width,
                'length_penalty': length_penalty,
                'add_BOS': False,
                'all_probs': False,
                'compute_logprobs': False,
            })
            
            # Add random seed if provided for reproducibility
            if random_seed is not None:
                gen_params['random_seed'] = random_seed
        
        # Use the simple generate method from TextGenerationModel
        response = self.model.generate(**gen_params)
        
        # Extract generated text
        generated_text = response[0]
        
        # Process the generated text to extract just the assistant's response
        # First, check if we have the full prompt returned
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
        
        # Remove the end token if present
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
        
        # Compute average for each metric
        evaluation_results = {}
        for metric in results:
            evaluation_results[metric] = np.mean(results[metric])
            
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
            
        print("Starting interactive chat with LLaMA 3 (type 'exit' to end)")
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


def prepare_chatbot_data(conversations, output_file):
    """
    Prepare conversation data for fine-tuning the chatbot.
    
    Args:
        conversations (list): List of conversation dictionaries.
        output_file (str): Path to save the processed data.
    """
    processed_data = []
    
    for conv in conversations:
        query = conv.get("user_message", "")
        answer = conv.get("assistant_message", "")
        
        if query and answer:
            processed_data.append({
                "query": query,
                "answer": answer
            })
    
    # Save to JSONL format
    with open(output_file, 'w') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Processed {len(processed_data)} conversations to {output_file}")


def generate_conversation_data(num_examples=100, output_file="training_conversations.jsonl"):
    """
    Generate synthetic conversation data for fine-tuning the chatbot.
    
    Args:
        num_examples (int): Number of conversation examples to generate.
        output_file (str): Path to save the generated data.
        
    Returns:
        list: The generated conversation data.
    """
    # Define topics and categories for diverse conversations
    topics = [
        "technology", "science", "health", "education", "entertainment",
        "travel", "food", "sports", "business", "arts", "history",
        "philosophy", "personal assistance", "general knowledge"
    ]
    
    # Various types of interactions
    interaction_types = [
        "question_answering", "information_seeking", "opinion_sharing",
        "problem_solving", "creative_requests", "casual_conversation"
    ]
    
    # Template questions and responses for each topic
    templates = {
        "technology": {
            "questions": [
                "Can you explain what {tech_concept} is?",
                "How does {tech_device} work?",
                "What's the difference between {tech_term1} and {tech_term2}?",
                "What are the latest developments in {tech_field}?",
                "Is {tech_product} worth buying?",
                "How can I troubleshoot {tech_issue}?",
                "What programming language is best for {tech_application}?",
                "What's your opinion on {tech_trend}?",
            ],
            "tech_concepts": ["blockchain", "cloud computing", "artificial intelligence", "machine learning", "virtual reality", "big data", "Internet of Things", "quantum computing", "edge computing"],
            "tech_devices": ["smartphone", "GPU", "SSD drives", "wireless earbuds", "smart home devices", "VR headsets", "fitness trackers", "wireless routers"],
            "tech_terms1": ["HTML", "RAM", "CPU", "OLED", "SaaS", "HTTP", "API", "SQL", "NoSQL"],
            "tech_terms2": ["CSS", "ROM", "GPU", "LCD", "PaaS", "HTTPS", "SDK", "MongoDB", "GraphDB"],
            "tech_fields": ["robotics", "cybersecurity", "web development", "mobile app development", "data science", "DevOps", "blockchain", "artificial intelligence"],
            "tech_products": ["iPhone 15", "Samsung Galaxy S23", "MacBook Pro", "Microsoft Surface", "AirPods Pro", "Oculus Quest", "Apple Watch", "Amazon Echo"],
            "tech_issues": ["slow Wi-Fi", "battery drain", "blue screen error", "software crashes", "system overheating", "data loss", "malware infection"],
            "tech_applications": ["web development", "data analysis", "mobile apps", "game development", "AI research", "automation", "cybersecurity"],
            "tech_trends": ["remote work technology", "AI ethics", "metaverse", "blockchain applications", "low-code platforms", "voice assistants", "autonomous vehicles"]
        },
        "science": {
            "questions": [
                "How does {science_process} work?",
                "Can you explain the theory of {science_theory}?",
                "What is the importance of {science_concept} in {science_field}?",
                "What are the latest discoveries about {science_topic}?",
                "How do scientists study {science_subject}?",
                "What's the difference between {science_term1} and {science_term2}?",
            ],
            "science_process": ["photosynthesis", "cellular respiration", "osmosis", "natural selection", "nuclear fusion", "DNA replication", "plate tectonics"],
            "science_theory": ["relativity", "quantum mechanics", "evolution", "big bang", "gravitational waves", "string theory", "dark matter"],
            "science_concept": ["entropy", "biodiversity", "momentum", "atomic structure", "symbiosis", "genetic engineering", "neural networks"],
            "science_field": ["astrophysics", "molecular biology", "ecology", "particle physics", "neuroscience", "climatology", "genetics"],
            "science_topic": ["black holes", "CRISPR", "exoplanets", "brain plasticity", "renewable energy", "antibiotic resistance", "quantum computing"],
            "science_subject": ["deep ocean ecosystems", "distant galaxies", "subatomic particles", "human genome", "climate patterns", "animal behavior", "chemical reactions"],
            "science_term1": ["gene", "hypothesis", "atom", "fusion", "photon", "theory", "asteroid"],
            "science_term2": ["chromosome", "theory", "molecule", "fission", "electron", "law", "meteor"]
        },
        "personal_assistance": {
            "questions": [
                "How can I become more productive?",
                "What are some tips for managing stress?",
                "How do I start a new habit?",
                "Can you help me plan a daily routine?",
                "What's a good way to organize my digital files?",
                "How can I improve my time management?",
                "What are effective ways to set and achieve goals?",
                "How can I maintain a work-life balance?",
                "What are some mindfulness techniques I can practice daily?",
                "How do I stay motivated when learning new skills?",
            ],
            "answers": [
                "To become more productive, try implementing the Pomodoro Technique where you work for 25 minutes and then take a 5-minute break. Also, prioritize your tasks using methods like the Eisenhower Matrix to focus on what's truly important rather than just urgent. Finally, minimize distractions by turning off notifications and designating specific times to check emails.",
                "Managing stress effectively involves several approaches: physical activity like walking or yoga can reduce stress hormones, deep breathing exercises can activate your parasympathetic nervous system, and practicing mindfulness helps you stay present rather than worrying about the future. Additionally, ensure you're getting adequate sleep and consider journaling to process difficult emotions.",
                "To start a new habit successfully, begin with a tiny version of it that takes less than two minutes to complete. Attach it to an existing habit (known as habit stacking), and make it obvious, attractive, easy, and satisfying as described in James Clear's Atomic Habits. Track your progress visually and allow at least 66 days for the habit to become automatic.",
                "A balanced daily routine might include: starting with a morning ritual (like hydration, movement, and planning), focused work during your peak energy hours, regular short breaks to maintain energy, dedicated time for learning and development, physical activity, social connection, and a wind-down routine in the evening to prepare for quality sleep.",
                "For organizing digital files, start with a consistent naming convention (date-category-description), create a logical folder hierarchy reflecting how you think about your projects, use cloud storage for accessibility and backup, regularly archive completed projects, and implement a systematic approach to email management like the PARA method (Projects, Areas, Resources, Archives).",
                "Improve time management by first tracking how you currently spend your time, then applying the 80/20 rule to focus on high-impact activities. Use time-blocking to schedule specific tasks, batch similar activities together, and set boundaries around your time including scheduled breaks. Learning to say no to low-priority requests is also essential.",
                "To effectively set and achieve goals, use the SMART framework (Specific, Measurable, Achievable, Relevant, Time-bound). Break large goals into smaller milestones, regularly review and adjust your approach, celebrate progress along the way, and consider accountability mechanisms like sharing goals with others or working with a coach.",
                "Maintaining work-life balance starts with clear boundaries – designate specific work hours and stick to them. Create transition rituals between work and personal time, prioritize non-negotiable personal activities on your calendar, regularly disconnect completely from work, and reassess your commitments periodically to ensure they align with your values and priorities.",
                "Daily mindfulness practices include: a 5-minute morning meditation focusing on your breath, mindful eating where you fully experience your food without distractions, a body scan technique to release tension, mindful walking paying attention to physical sensations, and the 5-4-3-2-1 grounding exercise using all your senses to connect with your environment.",
                "Stay motivated while learning by connecting the skill to meaningful personal goals, breaking learning into small achievable chunks, using spaced repetition rather than cramming, teaching concepts to others to reinforce understanding, celebrating small wins along the way, and finding a community of fellow learners for support and accountability."
            ]
        }
    }
    
    # General Q&A pairs
    general_qa_pairs = [
        {
            "user_message": "What's the weather like today?",
            "assistant_message": "I don't have access to real-time weather data. To get the current weather, you can check a weather website or app, or simply look outside your window!"
        },
        {
            "user_message": "Can you tell me a joke?",
            "assistant_message": "Why don't scientists trust atoms? Because they make up everything!"
        },
        {
            "user_message": "What's the meaning of life?",
            "assistant_message": "The meaning of life is a profound philosophical question that has been debated throughout human history. Some find meaning in relationships, personal growth, helping others, pursuing passions, or spiritual beliefs. Ultimately, many philosophers would argue that we each create our own meaning through the choices we make and the values we uphold."
        },
        {
            "user_message": "How can I learn programming?",
            "assistant_message": "Learning programming is a rewarding journey! Start by choosing a language suitable for beginners like Python or JavaScript. Use free resources like Codecademy, freeCodeCamp, or Khan Academy to learn the basics. Practice regularly with small projects that interest you. Join programming communities to ask questions and share your progress. Remember that consistency is key - even 30 minutes daily will help you make steady progress."
        },
        {
            "user_message": "Tell me about yourself.",
            "assistant_message": "I'm an AI assistant trained to be helpful, harmless, and honest. I can provide information, answer questions, assist with various tasks, and engage in conversations on a wide range of topics. I learn from interactions but don't have personal experiences or consciousness. My goal is to be a useful tool that helps you accomplish your goals while providing accurate and ethical assistance."
        },
        {
            "user_message": "What books do you recommend?",
            "assistant_message": "Book recommendations depend on your interests! Some widely acclaimed books across genres include: Fiction: 'To Kill a Mockingbird' by Harper Lee, '1984' by George Orwell. Non-fiction: 'Sapiens' by Yuval Noah Harari, 'Thinking, Fast and Slow' by Daniel Kahneman. Personal development: 'Atomic Habits' by James Clear. Science: 'A Brief History of Time' by Stephen Hawking. What genres interest you most? I can provide more targeted recommendations based on your preferences."
        },
        {
            "user_message": "How can I be more confident?",
            "assistant_message": "Building confidence takes time and practice: 1) Set small, achievable goals and celebrate when you reach them. 2) Practice positive self-talk by challenging negative thoughts. 3) Prepare thoroughly for challenging situations. 4) Adopt confident body language - stand tall with shoulders back. 5) Focus on your strengths while acknowledging areas for growth. 6) Step outside your comfort zone regularly with small challenges. 7) Accept that mistakes are valuable learning opportunities, not reflections of your worth."
        },
        {
            "user_message": "What's the best way to learn a new language?",
            "assistant_message": "The most effective language learning approach combines several elements: 1) Consistent daily practice (even 15 minutes). 2) Immersion through podcasts, shows, or music in your target language. 3) Conversation practice with native speakers or language exchange partners. 4) Spaced repetition for vocabulary using apps like Anki. 5) Focus on high-frequency words first. 6) Learning phrases rather than isolated words. 7) Setting specific goals like ordering food or having a 5-minute conversation. Remember that making mistakes is an essential part of the learning process!"
        },
        {
            "user_message": "How do I make friends as an adult?",
            "assistant_message": "Making friends as an adult takes intentional effort: 1) Pursue activities you genuinely enjoy - classes, clubs, volunteer work, or sports teams - to meet people with shared interests. 2) Be proactive in suggesting specific plans rather than vague intentions. 3) Practice being a good listener and showing authentic interest in others. 4) Maintain consistency through regular meetups. 5) Use apps like Meetup or Bumble BFF. 6) Reconnect with old friends or distant connections. 7) Be patient - meaningful friendships take time to develop through shared experiences."
        },
        {
            "user_message": "What are some good exercises for beginners?",
            "assistant_message": "Great beginner exercises include: 1) Walking - start with 15-30 minutes daily and gradually increase. 2) Body weight exercises like modified push-ups, squats, and lunges. 3) Swimming - gentle on joints while providing cardio and resistance. 4) Yoga or tai chi for flexibility, balance and core strength. 5) Stationary biking for low-impact cardio. Start any exercise program gradually, focus on proper form over intensity, schedule rest days, and listen to your body. Consider working with a trainer initially to learn correct techniques and prevent injury."
        }
    ]
    
    # Multi-turn conversations to include conversational context
    multi_turn_conversations = [
        [
            {
                "user_message": "I'm thinking about learning to code. Any suggestions on where to start?",
                "assistant_message": "That's an exciting decision! Programming is a valuable skill. For beginners, Python is often recommended because of its readable syntax and versatility. JavaScript is another good option if you're interested in web development. What are you hoping to do with coding once you learn it?"
            },
            {
                "user_message": "I'd like to analyze data and maybe build some simple applications.",
                "assistant_message": "For data analysis, Python would be an excellent choice! You could start with resources like Codecademy, DataCamp, or free courses on platforms like Coursera. The Python libraries pandas, NumPy, and matplotlib are essential for data analysis. For building applications, you might eventually want to explore frameworks like Flask or Django. Would you prefer interactive tutorials or more structured courses?"
            },
            {
                "user_message": "I think I learn better with structured courses. Are there any specific ones you'd recommend?",
                "assistant_message": "For structured learning with a focus on data analysis, I'd recommend: 1) 'Python for Everybody' specialization on Coursera by Dr. Charles Severance, 2) 'Data Science with Python' on edX, or 3) 'Python Data Science Handbook' by Jake VanderPlas if you prefer books. Start with basic Python concepts before diving into the data libraries. It's also helpful to work on small projects as you learn - perhaps analyzing a dataset you're personally interested in. This applied practice will reinforce what you're learning in the courses."
            }
        ],
        [
            {
                "user_message": "What's a good workout routine for someone who hasn't exercised in years?",
                "assistant_message": "Welcome back to fitness! It's important to start slowly and build up gradually. A good approach would be to begin with 2-3 days per week of mixed activity: 1) 15-20 minutes of walking or light cardio, 2) Basic strength movements like modified push-ups, chair squats, and standing dumbbell curls with very light weights, 3) Gentle stretching for major muscle groups. How does that sound as a starting point?"
            },
            {
                "user_message": "That seems doable. I'm worried about injuring myself though.",
                "assistant_message": "That's a valid concern! To minimize injury risk: 1) Always warm up for 5 minutes with light movement before any workout, 2) Focus on proper form rather than intensity or weight, 3) Start with fewer repetitions than you think you can do, 4) Increase difficulty gradually - no more than 10% per week, 5) Listen to your body - distinguish between muscle fatigue (normal) and pain (stop immediately), 6) Rest at least one day between strength sessions for the same muscle groups. Would you consider a session or two with a qualified trainer to learn proper form?"
            },
            {
                "user_message": "I'll look into that. How long before I should expect to see results?",
                "assistant_message": "Great question! You'll likely notice improvements in different areas at different times: 1) Mental benefits like improved mood and energy may appear within the first 1-2 weeks, 2) Endurance improvements often begin within 2-4 weeks, 3) Strength gains typically become noticeable within 4-6 weeks, 4) Visible physical changes usually take 6-8 weeks or longer, depending on consistency, nutrition, and individual factors. Remember that consistency matters more than intensity when you're starting out. Track your progress in multiple ways - not just weight or appearance, but also how far you can walk, how many repetitions you can complete, or how you feel after climbing stairs."
            }
        ]
    ]
    
    # Generate random conversation examples
    conversations = []
    
    # Add general Q&A pairs
    conversations.extend(general_qa_pairs)
    
    # Add some complete multi-turn conversations (flattened)
    for conversation in multi_turn_conversations:
        conversations.extend(conversation)
    
    # Generate topic-specific conversations
    for _ in range(num_examples - len(conversations)):
        topic = random.choice(topics)
        
        if topic in templates:
            topic_data = templates[topic]
            
            if "questions" in topic_data and len(topic_data["questions"]) > 0:
                question_template = random.choice(topic_data["questions"])
                
                # Fill in template variables if needed
                variables = {}
                for key in topic_data:
                    if key != "questions" and key != "answers" and isinstance(topic_data[key], list) and len(topic_data[key]) > 0:
                        variables[key] = random.choice(topic_data[key])
                
                question = question_template
                for var_name, var_value in variables.items():
                    placeholder = "{" + var_name + "}"
                    if placeholder in question:
                        question = question.replace(placeholder, var_value)
                
                # Generate or select an answer
                if "answers" in topic_data and len(topic_data["answers"]) > 0:
                    answer_template = random.choice(topic_data["answers"])
                    answer = answer_template
                    for var_name, var_value in variables.items():
                        placeholder = "{" + var_name + "}"
                        if placeholder in answer:
                            answer = answer.replace(placeholder, var_value)
                else:
                    # Generic answer if no templates available
                    answer = f"I'd be happy to discuss {topic} with you. {question} is an interesting question that involves considering several factors."
                
                conversations.append({
                    "user_message": question,
                    "assistant_message": answer
                })
            else:
                # Fallback for topics without templates
                conversations.append({
                    "user_message": f"Tell me about {topic}.",
                    "assistant_message": f"{topic.capitalize()} is a fascinating subject with many interesting aspects to explore. I'd be happy to discuss specific areas of {topic} that interest you."
                })
        else:
            # Generic conversation for topics without specific templates
            conversations.append({
                "user_message": f"What can you tell me about {topic}?",
                "assistant_message": f"{topic.capitalize()} is a broad and fascinating field. There are many aspects to explore, from fundamental concepts to practical applications. I'd be happy to discuss any specific aspects of {topic} that interest you."
            })
    
    # Save conversations to file
    prepare_chatbot_data(conversations, output_file)
    
    print(f"Generated {len(conversations)} conversation examples and saved to {output_file}")
    return conversations


def convert_hf_llama_to_nemo(hf_model_path, output_path):
    """
    Convert a HuggingFace LLaMA model to NeMo format.
    
    Args:
        hf_model_path (str): Path to HuggingFace model.
        output_path (str): Path to save the NeMo model.
    """
    try:
        from nemo.collections.nlp.modules.common.megatron.utils import convert_hf_checkpoint
        
        print(f"Converting HuggingFace model {hf_model_path} to NeMo format...")
        convert_hf_checkpoint(
            hf_model_path,
            output_path,
            model_class="gpt",
            hf_model_name="llama"
        )
        print(f"Model converted and saved to {output_path}")
    except Exception as e:
        print(f"Error converting model: {str(e)}")
        print("Make sure you have the latest nemo-toolkit with HuggingFace conversion support.")


def complete_workflow_example():
    """
    Complete workflow example for fine-tuning LLaMA 3 with NeMo.
    """
    print("===== Complete LLaMA 3 Fine-tuning Workflow =====")
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
    print("Learning rate:", finetuner.config.model.learning_rate)
    print("Batch size:", finetuner.config.model.global_batch_size)
    
    # Model preparation
    llama_path = input("\nEnter path to pretrained LLaMA 3 model (or press Enter to skip): ")
    if llama_path:
        print(f"\n3. Preparing model from {llama_path}...")
        finetuner.prepare_llama3_model(llama_path)
        
        print("\n4. Setting up training...")
        finetuner.setup_training()
        
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


def main():
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
    
    args = parser.parse_args()
    
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
        complete_workflow_example()
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
                args.interactive, args.evaluate, args.workflow]):
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


if __name__ == "__main__":
    main()
