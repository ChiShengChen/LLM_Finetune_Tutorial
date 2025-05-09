#!/usr/bin/env python3

import os
import random
import json

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


def main():
    """Generate conversation data for LLaMA 3 fine-tuning."""
    print("Generating synthetic conversation data for training...")
    
    # Generate training and validation data
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
    print("\nThese files can be used with NVIDIA NeMo to fine-tune a LLaMA 3 model.")


if __name__ == "__main__":
    main() 