# -*- coding: utf-8 -*-
import json
import re
import random
from typing import Optional, Dict, Any
from llama_cpp import Llama

class LLMEngine:
    def __init__(self, config_path):
        """Initialise le moteur LLM avec la configuration optimisée pour Mistral 7B"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        model_path = config.get('model_path', 'models/Mistral-Nemo-Instruct-2407.Q4_K_M.gguf')
        n_ctx = config.get('n_ctx', 4096)  # Contexte plus grand pour de meilleures performances
        n_threads = config.get('n_threads', 8)  # 8 threads pour le CPU i7-1165G7
        n_gpu_layers = config.get('n_gpu_layers', 4)  # Activer quelques couches GPU pour l'accélération
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,  # Activer l'accélération GPU partielle
            n_batch=512,  # Taille de lot optimisée pour la RAM disponible
            n_threads_batch=4,  # Threads par lot
            use_mmap=True,  # Utiliser mmap pour charger le modèle plus rapidement
            use_mlock=True,  # Verrouiller le modèle en mémoire pour de meilleures performances
            f16_kv=True,  # Utiliser float16 pour le cache KV
            vocab_only=False,
            verbose=False
        )
        
        # Paramètres de génération par défaut
        self.generation_params = {
            'temperature': config.get('temperature', 0.7),
            'top_p': config.get('top_p', 0.9),
            'top_k': config.get('top_k', 40),
            'repeat_penalty': config.get('repeat_penalty', 1.1),
            'max_tokens': config.get('max_tokens', 2048)
        }
    
    def ask(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None,
            top_p: Optional[float] = None, repeat_penalty: Optional[float] = None, 
            max_retries: int = 3) -> str:
        """
        Génère une réponse à partir d'un prompt avec des paramètres optimisés pour Mistral 7B
        
        Args:
            prompt: Le texte d'entrée
            max_tokens: Nombre maximum de tokens à générer (par défaut: depuis la config ou 2048)
            temperature: Contrôle la créativité (0.0-1.0, par défaut: depuis la config ou 0.7)
            top_p: Nucleus sampling (par défaut: depuis la config ou 0.9)
            repeat_penalty: Pénalité pour les répétitions (par défaut: depuis la config ou 1.1)
            max_retries: Nombre maximum de tentatives en cas d'échec (défaut: 3)
            
        Returns:
            str: La réponse générée
        """
        # Utiliser les valeurs par défaut de la configuration si non spécifiées
        params = self.generation_params.copy()
        if max_tokens is not None:
            params['max_tokens'] = min(max_tokens, 4096)  # Limiter à la taille maximale du contexte
        if temperature is not None:
            params['temperature'] = temperature
        if top_p is not None:
            params['top_p'] = top_p
        if repeat_penalty is not None:
            params['repeat_penalty'] = repeat_penalty

        # Liste des réponses de secours plus détaillées
        fallback_responses = [
            "Je ne suis pas tout à fait sûr de comprendre. Pourriez-vous fournir plus de contexte ou reformuler votre question ?",
            "Je veux m'assurer de bien comprendre votre question. Pourriez-vous fournir plus de détails ?",
            "Je m'efforce de fournir des réponses complètes. Pourriez essayer de formuler votre question différemment ?",
            "Pour vous fournir la meilleure réponse possible, pourriez-vous préciser votre demande ?"
        ]
        
        # Plusieurs tentatives en cas d'échec
        for attempt in range(max_retries):
            try:
                # Prépare les messages pour le chat
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Tu es Bissi, un expert dans de nombreux domaines. "
                            "Tu réponds de manière détaillée et professionnelle. "
                            "Utilise le format Markdown pour structurer tes réponses. "
                            "Sois concis mais complet dans tes explications. "
                            "Si tu ne sais pas, dis-le simplement."
                        )
                    },
                    {"role": "user", "content": prompt}
                ]
                
                # Calcul dynamique de la taille maximale du contexte
                context_size = self.llm.n_ctx()
                prompt_tokens = len(prompt.split())  # Estimation grossière
                max_possible_tokens = min(
                    params.get('max_tokens', 2048),
                    context_size - prompt_tokens - 100  # Marge de sécurité
                )
                
                # Appel au modèle avec des paramètres optimisés
                output = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_possible_tokens,
                    temperature=params['temperature'],
                    top_p=params['top_p'],
                    top_k=params.get('top_k', 40),
                    repeat_penalty=params['repeat_penalty'],
                    stop=["</s>", "<|endoftext|>", "User:", "Utilisateur:", "\n\n\n"],
                    stream=False
                )

                # Vérifie et traite la réponse
                if output and 'choices' in output and output['choices']:
                    response = output['choices'][0].get('message', {}).get('content', '').strip()
                    cleaned = self._clean_response(response)
                    
                    if cleaned and len(cleaned) > 5:  # Au moins 5 caractères
                        return cleaned
            
            except Exception as e:
                if attempt == max_retries - 1:  # Dernière tentative
                    print(f"Error generating response: {str(e)}")
                continue
        
        # Si on arrive ici, toutes les tentatives ont échoué
        return random.choice(fallback_responses)

    def _clean_response(self, text: str) -> str:
        """
        Nettoie la réponse générée en supprimant les éléments indésirables
        
        Args:
            text: Le texte brut à nettoyer
            
        Returns:
            str: Le texte nettoyé, ou une chaîne vide si le texte est invalide
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Supprime les préfixes de type "Bissi:" ou "Biissi:" (insensible à la casse)
        text = re.sub(r'^\s*(B+i+s+i+|Assistant|AI|Bot|A|Q)[:\s]*', '', text, flags=re.IGNORECASE)
        
        # Supprime les espaces en trop et les sauts de ligne multiples
        text = ' '.join(text.split())
        
        # Liste des préfixes à supprimer
        prefixes = [
            r'^[hH]ey[,!]?\s*',
            r'^[hH]i[,!]?\s*',
            r'^[hH]ello[,!]?\s*',
            r'^[oO]f\s+course[,!]?\s*',
            r'^[sS]ure[,!]?\s*',
            r'^[yY]es[,!]?\s*',
            r'^[nN]o[,!]?\s*',
            r'^[wW]ell[,!]?\s*',
            r'^[sS]o[,!]?\s*',
            r'^[aA]h[,!]?\s*',
            r'^[oO]h[,!]?\s*',
            r'^[uU]m[,!]?\s*',
            r'^[uU]h[,!]?\s*',
            r'^[lL]et\s+me\s+think[\s\.,!]*',
            r'^[lL]et\s+me\s+see[\s\.,!]*',
            r'^[iI]\'?m\s+not\s+sure[\s\.,!]*',
            r'^[iI]\s+think[\s\.,!]*',
            r'^[iI]\s+believe[\s\.,!]*',
            r'^[iI]\s+would\s+say[\s\.,!]*',
            r'^[tT]hat\'?s\s+a\s+good\s+question[\s\.,!]*',
            r'^[tT]hat\'?s\s+an\s+interesting\s+question[\s\.,!]*',
            r'^[aA]s\s+an?\s+ai[\s\.,!]*',
            r'^[aA]s\s+a\s+language\s+model[\s\.,!]*',
            r'^[aA]s\s+an?\s+ai\s+(assistant|language\s+model)[\s\.,!]*',
            r'^[aA]s\s+your\s+ai\s+assistant[\s\.,!]*',
            r'^[aA]s\s+an?\s+artificial\s+intelligence[\s\.,!]*',
            r'^[aA]s\s+an?\s+AI[\s\.,!]*',
            r'^[aA]s\s+an?\s+AI\s+language\s+model[\s\.,!]*',
            r'^[aA]s\s+an?\s+AI\s+assistant[\s\.,!]*',
            r'^[bB]ienvenue[\s\.,!]*',
            r'^[mM]on\s+ami[\s\.,!]*',
            r'^[jJ]\'ai\s+les\s+bonnes\s+mani[èe]res[\s\.,!]*',
            r'^[mM]erci\s+pour\s+votre\s+confiance[\s\.,!]*',
            r'^[jJ]e\s+suis\s+d[ée]sol[ée][\s\.,!]*',
            r'^[jJ]e\s+ne\s+comprends\s+pas[\s\.,!]*',
        ]
        
        # Supprime les préfixes indésirables
        for pattern in prefixes:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Nettoie les balises HTML/XML et caractères spéciaux
        text = re.sub(r'<[^>]+>', '', text)  # Balises HTML
        text = re.sub(r'[\r\n]+', ' ', text)  # Retours à la ligne
        text = re.sub(r'\s+', ' ', text)  # Espaces multiples
        text = text.strip(' .,;:!?\t\n\r\f\v\0\x0B')  # Espaces et ponctuation en début/fin
        
        # Supprime les points de suspension superflus
        text = re.sub(r'\.{3,}', '...', text)
        
        # Supprime les réponses vides ou trop courtes
        if len(text) < 10:  # Augmenté de 3 à 10 caractères minimum
            return ""
            
        # Capitalise la première lettre
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # S'assure que la réponse se termine par une ponctuation appropriée
        if text and text[-1] not in '.!?':
            # Vérifie si la dernière phrase est complète
            if any(text.rstrip().endswith(punc) for punc in [',', ';', ':', '-', '–', '—']):
                text = text.rstrip(' ,;:-–—') + '.'
            else:
                text += '.'
        
        # Détection de langues non-anglaises
        non_english_indicators = {
            'french': ['bonjour', 'salut', 'merci', 'au revoir', 'bienvenue', 's\'il vous plaît',
                     'je suis', 'comment ça va', 'ça va', 'pouvez-vous', 'pourriez-vous'],
            'spanish': ['hola', 'gracias', 'adiós', 'por favor', 'cómo estás', 'puedes', 'podrías'],
            'german': ['hallo', 'danke', 'auf wiedersehen', 'bitte', 'wie geht\'s', 'können sie'],
            'italian': ['ciao', 'grazie', 'arrivederci', 'per favore', 'come stai', 'puoi', 'potresti']
        }
        
        # Vérifie si la réponse contient des mots dans d'autres langues
        for lang, indicators in non_english_indicators.items():
            if any(indicator in text.lower() for indicator in indicators):
                return "I'm sorry, I can only respond in English. Could you please rephrase your question in English?"
        
        # Vérification des réponses tronquées
        if any(text.rstrip().endswith(cutoff) for cutoff in [',', ';', ':', ' -', '–', '—', '•']):
            # Si la réponse se termine par une ponctuation qui indique une coupure
            text = text.rstrip(' ,;:-–—•') + '.'
        
        # Vérification des listes incomplètes
        list_indicators = ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.',
                          '- ', '* ', '• ', '• ', '→ ', '> ']
        if any(text.rstrip().endswith(indicator) for indicator in list_indicators):
            # Supprime le dernier élément de liste incomplet
            lines = text.split('\n')
            if lines and any(lines[-1].strip().startswith(indicator) for indicator in list_indicators):
                text = '\n'.join(lines[:-1]).strip()
        
        # Gestion des questions sur la création ou la nature de Bissi
        creation_phrases = [
            'created', 'made', 'built', 'who are you', 'what are you', 
            'how were you', 'who made you', 'who built you', 'are you an ai',
            'are you a bot', 'are you human', 'what is your name', 'who created you'
        ]
        if any(phrase in text.lower() for phrase in creation_phrases):
            return "I'm here to help answer your questions. What would you like to know?"
            
        return text.strip()
        
    def __del__(self):
        """Nettoyage lors de la destruction de l'objet"""
        if hasattr(self, 'llm'):
            del self.llm