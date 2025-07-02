from sentence_transformers import SentenceTransformer
from claude import ClaudeLLM
from typing import List, Dict, Tuple
from datetime import datetime
import json
import re
from collections import Counter
import numpy as np
import os


class ConversationMemory:
    """Gère la mémoire de conversation du chatbot"""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_history: List[Dict] = []
        self.session_start = datetime.now()

    def add_exchange(self, question: str, answer: str, context: str = ""):
        """Ajoute un échange question-réponse à l'historique"""
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "context": context
        }

        self.conversation_history.append(exchange)

        # Limite la taille de l'historique
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    def get_recent_context(self, num_exchanges: int = 3) -> str:
        """Récupère le contexte des derniers échanges"""
        if not self.conversation_history:
            return ""

        recent_exchanges = self.conversation_history[-num_exchanges:]
        context_parts = []

        for exchange in recent_exchanges:
            context_parts.append(f"Q: {exchange['question']}")
            context_parts.append(f"R: {exchange['answer']}")

        return "\n".join(context_parts)

    def extract_context_keywords(self, num_exchanges: int = 5) -> List[str]:
        """Extrait les mots-clés importants de l'historique récent"""
        if not self.conversation_history:
            return []

        recent_exchanges = self.conversation_history[-num_exchanges:]
        text = " ".join([f"{ex['question']} {ex['answer']}" for ex in recent_exchanges])

        # Extraire les mots significatifs (plus de 3 caractères, pas de stop words basiques)
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]{4,}\b', text.lower())
        stop_words = {'in', 'with', 'to', 'but', 'who', 'what', 'how', 'why', 'this', 'that'}
        keywords = [word for word in words if word not in stop_words]

        # Retourner les mots les plus fréquents
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(10)]

    def get_full_history(self) -> List[Dict]:
        """Retourne l'historique complet"""
        return self.conversation_history.copy()

    def clear_history(self):
        """Efface l'historique de conversation"""
        self.conversation_history.clear()
        self.session_start = datetime.now()

    def save_to_file(self, filename: str):
        """Sauvegarde l'historique dans un fichier"""
        data = {
            "session_start": self.session_start.isoformat(),
            "conversation_history": self.conversation_history
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_file(self, filename: str):
        """Charge l'historique depuis un fichier"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.conversation_history = data.get("conversation_history", [])
                self.session_start = datetime.fromisoformat(data.get("session_start", datetime.now().isoformat()))
        except FileNotFoundError:
            print(f"Fichier {filename} non trouvé. Démarrage avec un historique vide.")


class JSONDocumentStore:
    """Gestionnaire de documents JSON avec recherche sémantique"""

    def __init__(self, json_file_path: str, embeddings_model: SentenceTransformer):
        self.json_file_path = json_file_path
        self.embeddings = embeddings_model
        self.documents = []
        self.document_embeddings = []
        self.load_documents()
        self.compute_embeddings()

    def load_documents(self):
        """Charge les documents depuis le fichier JSON"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            print(f"Chargé {len(self.documents)} documents depuis {self.json_file_path}")
        except FileNotFoundError:
            print(f"Fichier {self.json_file_path} non trouvé.")
            self.documents = []
        except json.JSONDecodeError as e:
            print(f"Erreur lors du décodage JSON: {e}")
            self.documents = []

    def compute_embeddings(self):
        """Calcule les embeddings pour tous les documents"""
        if not self.documents:
            self.document_embeddings = []
            return

        print("Calcul des embeddings pour les documents...")
        texts = [doc.get('text', '') for doc in self.documents]
        self.document_embeddings = self.embeddings.encode(texts)
        print(f"Embeddings calculés pour {len(self.document_embeddings)} documents")

    def similarity_search(self, query: str, k: int = 15) -> List[Dict]:
        """Recherche par similarité sémantique"""
        if not self.documents or len(self.document_embeddings) == 0:
            return []

        # Calculer l'embedding de la requête
        query_embedding = self.embeddings.encode([query])[0]

        # Calculer les similarités cosinus
        similarities = []
        for i, doc_embedding in enumerate(self.document_embeddings):
            similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((similarity, i))

        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Retourner les k documents les plus similaires
        results = []
        for similarity_score, doc_idx in similarities[:k]:
            doc = self.documents[doc_idx].copy()
            doc['similarity_score'] = similarity_score
            results.append(doc)

        return results


class Document:
    """Classe pour représenter un document avec métadonnées"""

    def __init__(self, page_content: str, metadata: Dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class TVATunisiaBot:
    """Chatbot spécialisé en TVA tunisienne avec base de données JSON et scoring hybride"""

    def __init__(self, json_file_path: str, max_history: int = 10):
        # Set up the embeddings
        self.embeddings = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        # Initialize conversation memory
        self.memory = ConversationMemory(max_history=max_history)

        # Set up JSON document store
        self.document_store = JSONDocumentStore(json_file_path, self.embeddings)

        # Initialize Claude LLM
        self.claude_llm = ClaudeLLM()

        # Configuration du scoring hybride
        self.hybrid_weights = {
            'semantic': 0.6,  # Poids pour la similarité sémantique
            'keyword': 0.3,  # Poids pour la correspondance par mots-clés
            'context': 0.1  # Poids pour la pertinence contextuelle
        }

        # Template avec contexte de conversation
        self.template = """our subject is microcloud lxd.
you are an assistant for questions concerning microcloud lxd.
CONVERSATION HISTORY:
{conversation_history}

CONTEXT :
{document_context}

INSTRUCTIONS:
use the context given to you and some of your knowledge to answer the questions.
if the subject gets out of the subject respond to the question anyway than remind him our subject.
when you use the context use naturally without mentioning it.


QUESTION: {question}

ANSWER:"""

    def _calculate_keyword_score(self, query: str, document_text: str) -> float:
        """Calcule le score de correspondance par mots-clés entre la requête et le document"""
        # Normaliser les textes
        query_words = set(re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', query.lower()))
        doc_words = set(re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', document_text.lower()))

        if not query_words:
            return 0.0

        # Calculer l'intersection
        common_words = query_words.intersection(doc_words)

        # Score basé sur le ratio de mots communs
        keyword_score = len(common_words) / len(query_words)

        # Bonus pour les mots exacts multiples
        if len(common_words) > 1:
            keyword_score *= 1.2

        return min(keyword_score, 1.0)

    def _calculate_context_score(self, document_text: str, context_keywords: List[str]) -> float:
        """Calcule le score de pertinence contextuelle basé sur l'historique"""
        if not context_keywords:
            return 0.0

        doc_text_lower = document_text.lower()
        matches = sum(1 for keyword in context_keywords if keyword in doc_text_lower)

        return matches / len(context_keywords)

    def _apply_hybrid_scoring(self, docs: List[Dict], query: str) -> List[Document]:
        """Applique le scoring hybride aux documents récupérés"""
        context_keywords = self.memory.extract_context_keywords()
        scored_docs = []

        for doc in docs:
            # Score sémantique (fourni par la recherche de similarité)
            semantic_score = doc.get('similarity_score', 0.0)

            # Score par mots-clés
            keyword_score = self._calculate_keyword_score(query, doc.get('text', ''))

            # Score contextuel
            context_score = self._calculate_context_score(doc.get('text', ''), context_keywords)

            # Calcul du score hybride pondéré
            hybrid_score = (
                    self.hybrid_weights['semantic'] * semantic_score +
                    self.hybrid_weights['keyword'] * keyword_score +
                    self.hybrid_weights['context'] * context_score
            )

            # Créer un objet Document avec les métadonnées
            document = Document(
                page_content=doc.get('text', ''),
                metadata={
                    'hybrid_score': hybrid_score,
                    'semantic_score': semantic_score,
                    'keyword_score': keyword_score,
                    'context_score': context_score,
                    'source_url': doc.get('source_url', ''),
                    'title': doc.get('title', ''),
                    'chunk_index': doc.get('chunk_index', 0),
                    'total_chunks': doc.get('total_chunks', 1)
                }
            )

            scored_docs.append(document)

        # Trier par score hybride décroissant
        scored_docs.sort(key=lambda x: x.metadata.get('hybrid_score', 0), reverse=True)

        return scored_docs

    def _select_diverse_chunks(self, docs: List[Document], max_chunks: int = 5, similarity_threshold: float = 0.85) -> \
    List[Document]:
        """Sélectionne des chunks diversifiés pour éviter la redondance"""
        if not docs:
            return []

        selected = [docs[0]]  # Prendre le meilleur document

        for doc in docs[1:]:
            if len(selected) >= max_chunks:
                break

            # Vérifier la similarité avec les documents déjà sélectionnés
            is_diverse = True
            doc_embedding = self.embeddings.encode([doc.page_content])[0]

            for selected_doc in selected:
                selected_embedding = self.embeddings.encode([selected_doc.page_content])[0]

                # Calculer la similarité cosinus
                similarity = np.dot(doc_embedding, selected_embedding) / (
                        np.linalg.norm(doc_embedding) * np.linalg.norm(selected_embedding)
                )

                if similarity > similarity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(doc)

        return selected

    def get_response(self, question: str) -> Tuple[str, str]:
        """Génère une réponse en utilisant le contexte, l'historique et le scoring hybride"""
        # Récupérer les documents pertinents depuis le JSON
        docs = self.document_store.similarity_search(question, k=5)

        # Appliquer le scoring hybride
        hybrid_scored_docs = self._apply_hybrid_scoring(docs, question)

        # Sélectionner des chunks diversifiés
        final_docs = self._select_diverse_chunks(hybrid_scored_docs, max_chunks=15)

        # Créer le contexte des documents
        document_context_parts = []
        for i, doc in enumerate(final_docs):
            # Ajouter le contenu du document
            document_context_parts.append(doc.page_content)

            # Optionnel - afficher les scores pour debug (commenté par défaut)
            # hybrid_score = doc.metadata.get('hybrid_score', 0)
            # print(f"Doc {i+1} - Score hybride: {hybrid_score:.3f}")

        document_context = "\n".join(document_context_parts)

        # Récupérer l'historique de conversation
        conversation_history = self.memory.get_recent_context(num_exchanges=3)

        # Créer le prompt avec contexte et historique
        prompt = self.template.format(
            conversation_history=conversation_history,
            document_context=document_context,
            question=question
        )

        # Générer la réponse
        response = self.claude_llm.generate_response(prompt)
        answer = response[0].text if hasattr(response[0], 'text') else str(response[0])

        # Ajouter l'échange à la mémoire
        self.memory.add_exchange(question, answer, document_context)

        return document_context, answer

    def reload_documents(self):
        """Recharge les documents depuis le fichier JSON"""
        print("Rechargement des documents...")
        self.document_store.load_documents()
        self.document_store.compute_embeddings()
        print("Documents rechargés avec succès.")

    def get_document_stats(self):
        """Affiche les statistiques des documents chargés"""
        print(f"\n=== STATISTIQUES DES DOCUMENTS ===")
        print(f"Nombre total de documents: {len(self.document_store.documents)}")
        print(f"Fichier JSON: {self.document_store.json_file_path}")
        if self.document_store.documents:
            print(f"Exemple de document:")
            example_doc = self.document_store.documents[0]
            print(f"  - Titre: {example_doc.get('title', 'N/A')}")
            print(f"  - URL: {example_doc.get('source_url', 'N/A')}")
            print(f"  - Longueur du texte: {len(example_doc.get('text', ''))}")
        print("=" * 35)

    def set_hybrid_weights(self, semantic: float = 0.6, keyword: float = 0.3, context: float = 0.1):
        """Permet d'ajuster les poids du scoring hybride"""
        total = semantic + keyword + context
        if abs(total - 1.0) > 0.01:  # Tolérance pour les erreurs de float
            print(f"Attention: La somme des poids ({total}) n'est pas égale à 1.0")

        self.hybrid_weights = {
            'semantic': semantic,
            'keyword': keyword,
            'context': context
        }
        print(f"Poids mis à jour: Sémantique={semantic}, Mots-clés={keyword}, Contexte={context}")

    def show_scoring_info(self):
        """Affiche les informations sur la configuration du scoring hybride"""
        print("\n=== CONFIGURATION SCORING HYBRIDE ===")
        print(f"Poids sémantique: {self.hybrid_weights['semantic']}")
        print(f"Poids mots-clés: {self.hybrid_weights['keyword']}")
        print(f"Poids contexte: {self.hybrid_weights['context']}")
        print(f"Nombre de documents dans la base: {len(self.document_store.documents)}")
        print("=" * 38)

    def show_history(self):
        """Affiche l'historique de conversation"""
        history = self.memory.get_full_history()
        if not history:
            print("Aucun historique de conversation.")
            return

        print("\n=== HISTORIQUE DE CONVERSATION ===")
        for i, exchange in enumerate(history, 1):
            print(f"\n[{i}] {exchange['timestamp']}")
            print(f"Q: {exchange['question']}")
            print(f"R: {exchange['answer'][:200]}..." if len(exchange['answer']) > 200 else f"R: {exchange['answer']}")
        print("=" * 35)

    def clear_memory(self):
        """Efface la mémoire de conversation"""
        self.memory.clear_history()
        print("Mémoire de conversation effacée.")

    def save_conversation(self, filename: str):
        """Sauvegarde la conversation"""
        self.memory.save_to_file(filename)
        print(f"Conversation sauvegardée dans {filename}")

    def load_conversation(self, filename: str):
        """Charge une conversation sauvegardée"""
        self.memory.load_from_file(filename)
        print(f"Conversation chargée depuis {filename}")


def main():
    """Fonction principale avec interface utilisateur améliorée"""
    # Chemin vers le fichier JSON des documents
    json_file_path = "ubuntu_microcloud_chunks1.json"  # Modifiez selon votre fichier

    # Vérifier si le fichier existe
    if not os.path.exists(json_file_path):
        print(f"Erreur: Le fichier {json_file_path} n'existe pas.")
        print("Veuillez créer un fichier JSON avec vos documents.")
        return

    bot = TVATunisiaBot(json_file_path)

    print("=== Chatbot Aurora avec Base de Données JSON ===")
    print("Commandes spéciales:")
    print("- 'quit' ou 'exit': quitter")
    print("- 'history': voir l'historique")
    print("- 'clear': effacer la mémoire")
    print("- 'save <nom_fichier>': sauvegarder la conversation")
    print("- 'load <nom_fichier>': charger une conversation")
    print("- 'scoring': voir la configuration du scoring hybride")
    print("- 'weights <sem> <key> <ctx>': ajuster les poids (ex: weights 0.5 0.4 0.1)")
    print("- 'stats': voir les statistiques des documents")
    print("- 'reload': recharger les documents depuis le JSON")
    print("=" * 55)

    # Afficher les statistiques au démarrage
    bot.get_document_stats()

    while True:
        question = input("\n🤖 Your question: ").strip()

        if question.lower() in ['quit', 'exit']:
            print("Bye Bye!")
            break

        elif question.lower() == 'history':
            bot.show_history()
            continue

        elif question.lower() == 'clear':
            bot.clear_memory()
            continue

        elif question.lower() == 'scoring':
            bot.show_scoring_info()
            continue

        elif question.lower() == 'stats':
            bot.get_document_stats()
            continue

        elif question.lower() == 'reload':
            bot.reload_documents()
            continue

        elif question.lower().startswith('weights '):
            try:
                parts = question.split()
                if len(parts) == 4:
                    sem, key, ctx = float(parts[1]), float(parts[2]), float(parts[3])
                    bot.set_hybrid_weights(sem, key, ctx)
                else:
                    print("Format: weights <sémantique> <mots-clés> <contexte>")
                    print("Exemple: weights 0.5 0.4 0.1")
            except ValueError:
                print("Erreur: Please use decimal numbers for weights.")
            continue

        elif question.lower().startswith('save '):
            filename = question[5:].strip()
            if filename:
                bot.save_conversation(filename)
            else:
                print("Specify file's name: save <nom_fichier>")
            continue

        elif question.lower().startswith('load '):
            filename = question[5:].strip()
            if filename:
                bot.load_conversation(filename)
            else:
                print("Specify file's name: load <file_name>")
            continue

        elif not question:
            continue

        try:
            context, response = bot.get_response(question)
            print(f"\n💬 Response: {response}")

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()