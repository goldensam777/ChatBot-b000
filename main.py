 # -*- coding: utf-8 -*-
# Bissi's chatbot mechanism starts here

import random
import re
from typing import List, Dict, Any, Optional, Tuple
from applib import *
from models.llm_engine import LLMEngine

class Bissi:
    default_name = "Bissi"
    
    greetings = {
        "startword": ["Bonjour", "Salut", "Bonsoir", "Coucou"],
        "question": [
            "Comment puis-je vous aider aujourd'hui ?",
            "Comment puis-je vous être utile ?",
            "Puis-je vous aider ?",
            "Que puis-je pour vous ?"
        ]
    }
    
    commands = {
        "quit": ["quit", "exit", "bye", "goodbye", "d"],
        "clear": ["clear", "reset", "new"],
        "help": ["help", "?", "commands"]
    }
    
    def __init__(self):
        self.name = self.default_name
        self.conversation_history = []
        
        # Initialise le moteur LLM
        print("Initialisation du modèle .gguf...")
        try:
            self.engine = LLMEngine("models/mistral7b_q4km_config.json")
            print("Le modèle a été chargé avec succès! \n")
        except Exception as e:
            print(f" Error loading model: {e}")
            exit(1)
        
        # Lance la conversation
        self.run()
    
    def greet(self) -> str:
        """Génère un message de bienvenue aléatoire en français"""
        startword = random.choice(self.greetings["startword"])
        question = random.choice(self.greetings["question"])
        return f"{startword} ! {question}"
    
    def to_user(self, text):
        """Affiche un message de Bissi"""
        print(f"{self.name}: {text}")
    
    def get_usr_ans(self) -> Optional[str]:
        """Récupère la réponse de l'utilisateur"""
        try:
            response = input("You: ").strip()
            return response if response else None
        except EOFError:  # Gère Ctrl+D
            return None
        except KeyboardInterrupt:  # Gère Ctrl+C
            print()  # Nouvelle ligne après ^C
            return None
    
    def show_help(self):
        """Affiche les commandes disponibles en français"""
        help_text = """
    Commandes disponibles :
  • quit/exit/bye/d - Quitter la conversation
  • clear/reset/new - Effacer l'historique
  • help/?/commands - Afficher cette aide
        """
        print(help_text)
    
    def is_command(self, user_input: str) -> tuple:
        """
        Vérifie si l'entrée est une commande
        Returns: (is_command, should_quit)
        """
        if not user_input:
            return (True, False)
        
        user_lower = user_input.lower()
        
        # Commande quit
        if user_lower in self.commands["quit"]:
            self.to_user("Au revoir ! Passez une excellente journée ! 👋")
            return (True, True)
        
        # Commande clear
        if user_lower in self.commands["clear"]:
            self.conversation_history = []
            self.to_user("Conversation effacée ! Recommençons depuis le début.")
            return (True, False)
        
        # Commande help
        if user_lower in self.commands["help"]:
            self.show_help()
            return (True, False)
        
        return (False, False)
    
    def format_context(self, history: List[Dict[str, str]], max_exchanges: int = 4) -> str:
        """Formate l'historique de conversation en une chaîne de caractères"""
        if not history:
            return ""
            
        # Prend les derniers échanges
        recent_history = history[-(max_exchanges * 2):]  # *2 car chaque échange a un message utilisateur et un assistant
        
        context_lines = []
        for msg in recent_history:
            role = "User" if msg['role'] == 'user' else self.name
            context_lines.append(f"{role}: {msg['content']}")
            
        return "\n".join(context_lines) + "\n"

    def clean_response(self, text: str) -> str:
        """Nettoie la réponse du modèle"""
        if not text:
            return ""
            
        # Supprime les balises HTML/XML
        text = re.sub(r'<[^>]+>', '', text)
        
        # Supprime les espaces en trop
        text = ' '.join(text.split())
        
        # Supprime les guillemets superflus
        text = text.strip('"\'')
        
        return text.strip()

    def generate_response(self, user_input: str) -> str:
        """Génère une réponse en utilisant le LLM"""
        try:
            # Vérifie d'abord si c'est une salutation simple
            user_input_lower = user_input.lower().strip()
            greetings = ['hello', 'hi', 'hey', 'greetings', 'bonjour', 'salut']
            
            if any(greeting in user_input_lower for greeting in greetings):
                return random.choice([
                    "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
                    "Salut ! Comment puis-je vous être utile ?",
                    "Bonsoir ! En quoi puis-je vous aider ?"
                ])
            
            # Formate le contexte de conversation (garde 4 derniers échanges)
            context = self.format_context(self.conversation_history, max_exchanges=4)
            
            # Construit le prompt complet avec des instructions claires
            system_prompt = """Tu es Bissi, une IA d'assistance multilingue avec le français comme langue principale.
            - Réponds principalement en français, sauf si on te demande une autre langue.
            - Sois clair, concis et informatif dans tes réponses.
            - Si tu ne sais pas quelque chose, dis-le simplement.
            - Sois amical et professionnel dans tes réponses.
            - Tu peux répondre dans d'autres langues si on te le demande.
            
            Conversation en cours :
            """
            
            if context:
                full_prompt = f"{system_prompt}{context}User: {user_input}"
            else:
                full_prompt = f"{system_prompt}User: {user_input}"
            
            # Génère la réponse via le LLM avec des paramètres plus stricts
            response = self.engine.ask(
                prompt=full_prompt,
                max_tokens=1000,
                temperature=0.7,  # Un peu plus déterministe
                top_p=0.9,
                repeat_penalty=1.2
            )
            
            # Nettoie la réponse
            response = self.clean_response(response)
            
            # Validation de la réponse
            if not response or len(response.strip()) < 2:
                return "Je ne suis pas sûr de comprendre. Pourriez-vous reformuler ou fournir plus de détails ?"
            
            # Vérifie si la réponse est dans une langue étrange
            if self._is_gibberish(response):
                return "Je m'excuse, mais j'ai du mal à générer une réponse appropriée. Pourriez-vous reformuler votre question en français ou en anglais ?"
            
            # Coupe la réponse si elle est trop longue
            if len(response) > 500:
                response = response[:497] + '...'
                
            return response
            
        except Exception as e:
            print(f"⚠️ Error generating response: {e}")
            return "Désolé, une erreur s'est produite. Veuillez réessayer."
    
    def _is_gibberish(self, text: str) -> bool:
        """Detects if the text is incoherent or in an unsupported language"""
        if not text or len(text.strip()) < 2:
            return True
            
        # Common words in multiple languages that indicate valid responses
        valid_words = [
            # English
            'hello', 'hi', 'hey', 'greetings', 'help', 'thanks', 'thank you', 
            'ok', 'yes', 'no', 'goodbye', 'bye', 'clear', 'reset', 'new',
            'what', 'how', 'when', 'where', 'why', 'who', 'which',
            'can', 'could', 'would', 'should', 'will', 'is', 'are', 'am',
            
            # French
            'bonjour', 'salut', 'merci', 'au revoir', 'comment ça va',
            'oui', 'non', 'peut-être', 'pourquoi', 'comment', 'quand',
            'où', 'qui', 'quoi', 'd''accord', 'bien', 'mal',
            
            # Spanish
            'hola', 'adiós', 'gracias', 'por favor', 'sí', 'no',
            'cómo', 'cuándo', 'dónde', 'por qué', 'qué', 'quién',
            
            # Common AI/tech terms
            'ai', 'artificial intelligence', 'machine learning', 'neural network',
            'data', 'algorithm', 'computer', 'programming', 'code', 'model'
        ]
        
        text_lower = text.lower()
        
        # If the text contains any valid words, it's probably fine
        if any(word in text_lower for word in valid_words):
            return False
        
        # Check for very short or very long responses
        if len(text) < 3 or len(text) > 1000:  # Increased max length
            return True
            
        # Check for excessive non-ASCII characters (but be more lenient)
        non_ascii_ratio = sum(1 for char in text if ord(char) > 127) / len(text)
        if non_ascii_ratio > 0.5:  # Increased threshold to 50%
            return True
            
        # If we get here, the text is probably fine
        return False
    
    def save_exchange(self, user_input: str, bot_response: str):
        """Sauvegarde l'échange dans l'historique"""
        self.conversation_history.append({
            'role': 'user',
            'content': user_input
        })
        self.conversation_history.append({
            'role': 'assistant',
            'content': bot_response
        })
        
        # Limite l'historique à 16 messages (8 échanges)
        if len(self.conversation_history) > 16:
            self.conversation_history = self.conversation_history[-16:]
    
    def run(self):
        """Boucle principale de conversation en français"""
        print("=" * 60)
        print(f"{self.name} - Assistant IA Multilingue")
        print("💡 Astuce: Tapez 'help' pour les commandes")
        print("=" * 60)
        print()
        
        # Message de bienvenue
        greeting_text = self.greet()
        self.to_user(greeting_text)
        print()
        
        # Boucle de conversation
        conversation_active = True
        while conversation_active:
            try:
                # Récupère l'entrée utilisateur
                answer = self.get_usr_ans()
                
                # Vérifie si l'utilisateur veut quitter (None = Ctrl+D ou 'd')
                if answer is None:
                    self.to_user("Au revoir ! 👋")
                    break
                
                # Vérifie si c'est une commande
                is_cmd, should_quit = self.is_command(answer)
                
                if should_quit:
                    break
                
                if is_cmd:
                    print()
                    continue
                
                # Génère et affiche la réponse
                response = self.generate_response(answer)
                self.to_user(response)
                
                # Sauvegarde l'échange
                self.save_exchange(answer, response)
                print()
                
            except KeyboardInterrupt:
                print("\n")
                self.to_user("Interrompu ! Au revoir ! 👋")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}\n")

# Point d'entrée
if __name__ == "__main__":
    bot = Bissi()