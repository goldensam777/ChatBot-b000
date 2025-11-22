# -*- coding: utf-8 -*-
import random

# Mécanismes pour chatbot.py

def parse_str(question=None) -> str:
    """Récupère l'entrée utilisateur avec validation"""
    var = ""
    while var == "":
        try:
            var = input(question + " : ").strip()
            if var.lower() == "d":
                return None
        except (KeyboardInterrupt, EOFError):
            return None
    return var

def randomization(array: list):
    """Sélectionne un élément aléatoire dans une liste"""
    return random.choice(array)

def clean_response(text: str) -> str:
    """Nettoie le texte de réponse du LLM"""
    if not text:
        return ""
    
    # Retire les préfixes indésirables
    prefixes = ["Bissi:", "Assistant:", "AI:", "Bot:", "Sure!", "Sure,"]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    # Retire les espaces multiples
    text = " ".join(text.split())
    
    return text

def format_context(history: list, max_exchanges: int = 5) -> str:
    """Formate l'historique de conversation pour le LLM"""
    context = ""
    for msg in history[-max_exchanges:]:
        if msg['role'] == 'user':
            context += f"User: {msg['content']}\n"
        else:
            context += f"Bissi: {msg['content']}\n"
    return context