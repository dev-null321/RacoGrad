#!/usr/bin/env python3
"""
CCL Data: Separate train/eval templates for genuine generalization testing.
"""

import torch
from typing import List, Dict


SKILL_NAMES = ["code", "qa", "summarize", "translate"]


# ============================================================
# TRAIN templates (used during training only)
# ============================================================

TRAIN_TEMPLATES = {
    "code": [
        "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\n",
        "def sort_list(lst):\n    return sorted(lst)\n",
        "def find_max(nums):\n    return max(nums)\n",
        "def reverse_str(s):\n    return s[::-1]\n",
        "def count_words(text):\n    return len(text.split())\n",
        "def is_palindrome(s):\n    return s == s[::-1]\n",
        "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a+b\n    return a\n",
        "def merge_dicts(d1, d2):\n    return {**d1, **d2}\n",
        "def flatten(lst):\n    return [x for sub in lst for x in sub]\n",
        "def gcd(a, b):\n    while b: a, b = b, a%b\n    return a\n",
        "def is_even(n):\n    return n % 2 == 0\n",
        "def square(x):\n    return x * x\n",
        "def abs_val(x):\n    return x if x >= 0 else -x\n",
        "def char_count(s, c):\n    return s.count(c)\n",
        "def list_sum(lst):\n    return sum(lst)\n",
    ],
    "qa": [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
        ("What is photosynthesis?", "Photosynthesis is the process by which plants convert sunlight into energy."),
        ("How many planets are in our solar system?", "There are 8 planets in our solar system."),
        ("What is the speed of light?", "The speed of light is approximately 299,792,458 meters per second."),
        ("What is DNA?", "DNA is a molecule that carries genetic instructions for life."),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
        ("What is gravity?", "Gravity is a fundamental force that attracts objects with mass toward each other."),
        ("When did World War II end?", "World War II ended in 1945."),
        ("What is the largest ocean?", "The Pacific Ocean is the largest ocean on Earth."),
        ("What is the boiling point of water?", "Water boils at 100 degrees Celsius at sea level."),
        ("Who discovered penicillin?", "Alexander Fleming discovered penicillin in 1928."),
        ("What is the tallest mountain?", "Mount Everest is the tallest mountain above sea level."),
        ("What causes tides?", "Tides are primarily caused by the gravitational pull of the Moon."),
        ("What is an atom?", "An atom is the smallest unit of ordinary matter."),
    ],
    "summarize": [
        ("The Amazon rainforest covers approximately 5.5 million square kilometers and is home to about 10 percent of all species on Earth. It plays a crucial role in regulating the global climate.",
         "The Amazon is a vast, biodiverse rainforest critical to global climate regulation."),
        ("Artificial intelligence has made remarkable progress in recent years, particularly with large language models that can generate text, write code, and answer questions with increasing sophistication.",
         "AI has advanced significantly, especially through capable large language models."),
        ("The Great Wall of China stretches over 21,000 kilometers and was built over many centuries to protect Chinese states from nomadic invasions from the north.",
         "The Great Wall is a massive ancient fortification spanning over 21,000 km."),
        ("Climate change is causing more frequent and severe weather events worldwide, including hurricanes, droughts, and flooding, affecting millions of people annually.",
         "Climate change intensifies extreme weather events globally, impacting millions."),
        ("The human brain contains approximately 86 billion neurons, each connected to thousands of others, forming the most complex known structure in the universe.",
         "The brain is an extraordinarily complex organ with 86 billion interconnected neurons."),
        ("Ocean acidification caused by increased CO2 absorption threatens marine ecosystems, particularly coral reefs and shellfish that depend on calcium carbonate structures.",
         "Rising CO2 levels acidify oceans, endangering coral reefs and shellfish."),
        ("The invention of the printing press by Gutenberg around 1440 revolutionized the spread of knowledge and is considered one of the most important innovations in human history.",
         "Gutenberg's printing press transformed knowledge dissemination worldwide."),
        ("Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot efficiently replicate.",
         "Quantum computing leverages quantum physics for fundamentally new computation."),
    ],
    "translate": [
        ("Hello, how are you?", "Bonjour, comment allez-vous?"),
        ("The cat is on the table.", "Le chat est sur la table."),
        ("I love programming.", "J'adore la programmation."),
        ("What time is it?", "Quelle heure est-il?"),
        ("Good morning.", "Bonjour."),
        ("Thank you very much.", "Merci beaucoup."),
        ("The weather is nice today.", "Il fait beau aujourd'hui."),
        ("See you tomorrow.", "A demain."),
        ("I am happy.", "Je suis heureux."),
        ("Good night.", "Bonne nuit."),
        ("Where is the train station?", "Où est la gare?"),
        ("I would like some water.", "Je voudrais de l'eau."),
        ("The book is interesting.", "Le livre est intéressant."),
        ("She speaks three languages.", "Elle parle trois langues."),
        ("We are going to the park.", "Nous allons au parc."),
    ],
}


# ============================================================
# EVAL templates (completely different from train)
# ============================================================

EVAL_TEMPLATES = {
    "code": [
        "def power(base, exp):\n    return base ** exp\n",
        "def unique(lst):\n    return list(set(lst))\n",
        "def min_val(nums):\n    return min(nums)\n",
        "def join_str(lst, sep):\n    return sep.join(lst)\n",
        "def is_prime(n):\n    return n > 1 and all(n % i for i in range(2, int(n**0.5)+1))\n",
        "def zip_lists(a, b):\n    return list(zip(a, b))\n",
        "def dict_keys(d):\n    return list(d.keys())\n",
        "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n",
        "def remove_dupes(lst):\n    seen = set()\n    return [x for x in lst if x not in seen and not seen.add(x)]\n",
        "def prod(lst):\n    r = 1\n    for x in lst: r *= x\n    return r\n",
    ],
    "qa": [
        ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
        ("Who invented the telephone?", "Alexander Graham Bell invented the telephone."),
        ("What is the smallest country in the world?", "Vatican City is the smallest country in the world."),
        ("How many bones are in the human body?", "An adult human body has 206 bones."),
        ("What year was the internet invented?", "The internet originated from ARPANET in 1969."),
        ("What is the hardest natural substance?", "Diamond is the hardest natural substance."),
        ("Who wrote Pride and Prejudice?", "Jane Austen wrote Pride and Prejudice."),
        ("What is the largest planet?", "Jupiter is the largest planet in our solar system."),
        ("What causes earthquakes?", "Earthquakes are caused by tectonic plate movements."),
        ("What is the deepest ocean trench?", "The Mariana Trench is the deepest ocean trench."),
    ],
    "summarize": [
        ("Renewable energy sources like solar and wind power have seen dramatic cost reductions over the past decade, making them increasingly competitive with fossil fuels in many markets around the world.",
         "Renewable energy costs have dropped sharply, competing with fossil fuels globally."),
        ("The discovery of antibiotics in the 20th century transformed medicine by providing effective treatments for bacterial infections that were previously often fatal.",
         "Antibiotics revolutionized medicine by enabling treatment of deadly bacterial infections."),
        ("Space exploration has yielded numerous technological spinoffs that benefit everyday life, from memory foam to water purification systems and satellite communications.",
         "Space exploration has produced many practical technologies used in daily life."),
        ("Deforestation in tropical regions releases stored carbon dioxide and destroys habitats for countless species, contributing to both climate change and biodiversity loss.",
         "Tropical deforestation drives climate change and destroys biodiversity."),
        ("The development of vaccines has been one of public health's greatest achievements, preventing millions of deaths from diseases like smallpox, polio, and measles.",
         "Vaccines have saved millions of lives by preventing deadly infectious diseases."),
    ],
    "translate": [
        ("The sun is shining brightly.", "Le soleil brille intensément."),
        ("I need to buy groceries.", "J'ai besoin d'acheter des courses."),
        ("The children are playing outside.", "Les enfants jouent dehors."),
        ("Can you help me please?", "Pouvez-vous m'aider s'il vous plaît?"),
        ("The restaurant is closed today.", "Le restaurant est fermé aujourd'hui."),
        ("I enjoy reading books.", "J'aime lire des livres."),
        ("The movie was excellent.", "Le film était excellent."),
        ("Winter is my favorite season.", "L'hiver est ma saison préférée."),
        ("He works at the hospital.", "Il travaille à l'hôpital."),
        ("The flowers are beautiful.", "Les fleurs sont belles."),
    ],
}


def generate_dataset(tokenizer, skill_name: str, split: str = "train",
                     num_samples: int = 200, seq_len: int = 128) -> List[Dict]:
    """Generate dataset using split-appropriate templates."""
    templates = TRAIN_TEMPLATES[skill_name] if split == "train" else EVAL_TEMPLATES[skill_name]
    data = []

    for i in range(num_samples):
        t = templates[i % len(templates)]
        if skill_name == "code":
            text = f"# Python function\n{t}"
        elif skill_name == "qa":
            q, a = t
            text = f"Question: {q}\nAnswer: {a}\n"
        elif skill_name == "summarize":
            p, s = t
            text = f"Passage: {p}\nSummary: {s}\n"
        elif skill_name == "translate":
            en, fr = t
            text = f"Translate English to French:\nEnglish: {en}\nFrench: {fr}\n"
        data.append({"text": text})

    return data


def tokenize_dataset(data: List[Dict], tokenizer, seq_len: int) -> torch.Tensor:
    all_ids = []
    for item in data:
        ids = tokenizer.encode(item["text"], add_special_tokens=True)
        if len(ids) >= seq_len:
            ids = ids[:seq_len]
        else:
            ids = ids + [tokenizer.eos_token_id] * (seq_len - len(ids))
        all_ids.append(ids)
    return torch.tensor(all_ids, dtype=torch.long)
