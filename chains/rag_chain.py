from sentence_transformers import SentenceTransformer

from claude import ClaudeLLM
from typing import List, Dict, Tuple
from datetime import datetime
import json
import re  
from collections import Counter  
import numpy as np  
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from settings import OpenSearchSettings
load_dotenv()
# Initialize OpenSearch settings
opensearch_config = OpenSearchSettings()
VECTOR_DIM = 768  # Should match the embedding model output
OPENSEARCH_URL = f"http://{opensearch_config.host}:{opensearch_config.port}"

class ConversationMemory:
    """Manages conversation history with context extraction and keyword analysis"""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_history: List[Dict] = []
        self.session_start = datetime.now()

    def add_exchange(self, question: str, answer: str, context: str = ""):
        """Adds a question-answer exchange to the conversation history"""
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
        """Retrieves the context of recent exchanges"""
        if not self.conversation_history:
            return ""

        recent_exchanges = self.conversation_history[-num_exchanges:]
        context_parts = []

        for exchange in recent_exchanges:
            context_parts.append(f"Q: {exchange['question']}")
            context_parts.append(f"R: {exchange['answer']}")

        return "\n".join(context_parts)

    # AJOUT: Nouvelle méthode pour extraire les mots-clés du contexte conversationnel
    def extract_context_keywords(self, num_exchanges: int = 5) -> List[str]:
        """Extracts important keywords from recent conversation history"""
        if not self.conversation_history:
            return []

        recent_exchanges = self.conversation_history[-num_exchanges:]
        text = " ".join([f"{ex['question']} {ex['answer']}" for ex in recent_exchanges])

        # Extract significant words (more than 3 characters, no basic stop words)
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]{4,}\b', text.lower())
        stop_words = {'dans', 'avec', 'pour', 'mais', 'que', 'qui', 'quoi', 'comment', 'pourquoi', 'cette', 'cette'}
        keywords = [word for word in words if word not in stop_words]

        # Return the most frequent words
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(10)]

    def get_full_history(self) -> List[Dict]:
        """Returns the full conversation history"""
        return self.conversation_history.copy()

    def clear_history(self):
        """Clears the conversation history"""
        self.conversation_history.clear()
        self.session_start = datetime.now()

    def save_to_file(self, filename: str):
        """Saves the conversation history to a file"""
        data = {
            "session_start": self.session_start.isoformat(),
            "conversation_history": self.conversation_history
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_file(self, filename: str):
        """Loads conversation history from a file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.conversation_history = data.get("conversation_history", [])
                self.session_start = datetime.fromisoformat(data.get("session_start", datetime.now().isoformat()))
        except FileNotFoundError:
            print(f"File {filename} not found. Starting with empty history.")


class AuroraBot:

    def __init__(self,  max_history: int = 10):
        # Set up the embeddings
        self.embedding_model = HuggingFaceEmbeddings(model_name=opensearch_config.embedding_model_name)
        self.embeddings = SentenceTransformer(opensearch_config.embedding_model_name)

        # Initialize conversation memory
        self.memory = ConversationMemory(max_history=max_history)

        # Set up Qdrant client
# Set up OpenSearch client
        self.opensearch_client = OpenSearch(
            hosts=[{'host': opensearch_config.host, 'port': opensearch_config.port}],
            http_auth=(opensearch_config.user, opensearch_config.password.get_secret_value()),  # Or use AWS/IAM auth as needed
            use_ssl=opensearch_config.use_ssl,
            verify_certs=opensearch_config.verify_certs  # ⚠️ Only for testing/dev
        )
        # Set up vector store
        self.vectorstore = OpenSearchVectorSearch(
            opensearch_url=OPENSEARCH_URL,
            client=self.opensearch_client,
            index_name= opensearch_config.index_name,
            embedding_function=self.embedding_model
        )

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})  

        # Initialize Claude LLM
        self.claude_llm = ClaudeLLM()

        # Initialize hybrid scoring weights
        self.hybrid_weights = {
            'semantic': 0.6,  # Weight for semantic similarity
            'keyword': 0.3,  # Weight for keyword matching
            'context': 0.1  # Weight for contextual relevance
        }

        # Template with conversation context


        

        self.template = """our subject is microcloud lxd.
        you are an assistant for questions concerning microcloud lxd.
        CONVERSATION HISTORY:
        {conversation_history}

        CONTEXT:
        {document_context}

        INSTRUCTIONS:
        use the context given to you and some of your knowledge to answer the questions.
        if the subject gets out of the subject respond to the question anyway than remind him our subject.
        when you use the context use it naturally without mentioning it.


        QUESTION: {question}

        ANSWER:"""

    def _embedding_function(self, text):
        """Embedding function for OpenSearch"""
        if isinstance(text, str):
            text = [text]
        return self.embeddings.encode(text)[0].tolist()

    def _calculate_keyword_score(self, query: str, document_text: str) -> float:
        """Calculates the keyword matching score between the query and the document"""
        # Normalize the texts
        query_words = set(re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', query.lower()))
        doc_words = set(re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', document_text.lower()))

        if not query_words:
            return 0.0

        # Calculate intersection
        common_words = query_words.intersection(doc_words)

        # Score based on the ratio of common words
        keyword_score = len(common_words) / len(query_words)

        # Bonus for exact matching words
        if len(common_words) > 1:
            keyword_score *= 1.2

        return min(keyword_score, 1.0)

    def _calculate_context_score(self, document_text: str, context_keywords: List[str]) -> float:
        """Calculates the contextual relevance score based on history"""
        if not context_keywords:
            return 0.0

        doc_text_lower = document_text.lower()
        matches = sum(1 for keyword in context_keywords if keyword in doc_text_lower)

        return matches / len(context_keywords)

    # Initialize hybrid scoring
    def _apply_hybrid_scoring(self, docs, query: str) -> List:
        """Applies hybrid scoring to retrieved documents"""
        context_keywords = self.memory.extract_context_keywords()
        scored_docs = []

        for doc in docs:
            # Semantic score (provided by Qdrant via similarity_search_with_score)
            semantic_score = getattr(doc, 'metadata', {}).get('score', 0.8)  # Default score if not available

            # Keyword matching score
            keyword_score = self._calculate_keyword_score(query, doc.page_content)

            # Contextual score
            context_score = self._calculate_context_score(doc.page_content, context_keywords)

            # Calculate weighted hybrid score
            hybrid_score = (
                    self.hybrid_weights['semantic'] * semantic_score +
                    self.hybrid_weights['keyword'] * keyword_score +
                    self.hybrid_weights['context'] * context_score
            )

            # Store the score in the metadata
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            doc.metadata['hybrid_score'] = hybrid_score
            doc.metadata['semantic_score'] = semantic_score
            doc.metadata['keyword_score'] = keyword_score
            doc.metadata['context_score'] = context_score

            scored_docs.append(doc)

        # Sort by hybrid score descending
        scored_docs.sort(key=lambda x: x.metadata.get('hybrid_score', 0), reverse=True)

        return scored_docs

    # ADD: Method to select the best chunks with diversity
    def _select_diverse_chunks(self, docs, max_chunks: int = 5, similarity_threshold: float = 0.85) -> List:
        """Selects diverse chunks to avoid redundancy"""
        if not docs:
            return []

        selected = [docs[0]]  # Take the best document

        for doc in docs[1:]:
            if len(selected) >= max_chunks:
                break

            # Check similarity with already selected documents
            is_diverse = True
            doc_embedding = self.embeddings.encode([doc.page_content])[0]

            for selected_doc in selected:
                selected_embedding = self.embeddings.encode([selected_doc.page_content])[0]

                # Calculate cosine similarity
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
        """Generates a response using context, history, and hybrid scoring"""
        # Retrieve relevant documents (more documents for scoring)
        docs = self.retriever.invoke(question)

        # ADD: Apply hybrid scoring
        hybrid_scored_docs = self._apply_hybrid_scoring(docs, question)

        # ADD: Select diverse chunks
        final_docs = self._select_diverse_chunks(hybrid_scored_docs, max_chunks=5)

        # Create document context with scoring information (optional for debug)
        document_context_parts = []
        for i, doc in enumerate(final_docs):
            # Add document content
            document_context_parts.append(doc.page_content)

            # ADD: Optional - display scores for debug (commented out by default)
            # hybrid_score = doc.metadata.get('hybrid_score', 0)
            # print(f"Doc {i+1} - Hybrid score: {hybrid_score:.3f}")

        document_context = "\n".join(document_context_parts)

        # Retrieve conversation history
        conversation_history = self.memory.get_recent_context(num_exchanges=3)

        # Create prompt with context and history
        prompt = self.template.format(
            conversation_history=conversation_history,
            document_context=document_context,
            question=question
        )

        # Generate response
        response = self.claude_llm.generate_response(prompt)
        answer = response[0].text if hasattr(response[0], 'text') else str(response[0])

        # Add exchange to memory
        self.memory.add_exchange(question, answer, document_context)

        return document_context, answer

    # ADD: Method to adjust hybrid scoring weights
    def set_hybrid_weights(self, semantic: float = 0.6, keyword: float = 0.3, context: float = 0.1):
        """Allows adjusting hybrid scoring weights"""
        total = semantic + keyword + context
        if abs(total - 1.0) > 0.01:  # Tolerance for float errors
            print(f"Warning: The sum of weights ({total}) is not equal to 1.0")

        self.hybrid_weights = {
            'semantic': semantic,
            'keyword': keyword,
            'context': context
        }
        print(f"Poids mis à jour: Sémantique={semantic}, Mots-clés={keyword}, Contexte={context}")

    # ADD: Method to display hybrid scoring information
    def show_scoring_info(self):
        """Displays information about the hybrid scoring configuration"""
        print("\n=== HYBRID SCORING CONFIGURATION ===")
        print(f"Semantic weight: {self.hybrid_weights['semantic']}")
        print(f"Keyword weight: {self.hybrid_weights['keyword']}")
        print(f"Context weight: {self.hybrid_weights['context']}")
        print(f"Number of retrieved documents: {self.retriever.search_kwargs['k']}")
        print("=" * 38)

    def show_history(self):
        """Displays conversation history"""
        history = self.memory.get_full_history()
        if not history:
            print("No conversation history.")
            return

        print("\n=== CONVERSATION HISTORY ===")
        for i, exchange in enumerate(history, 1):
            print(f"\n[{i}] {exchange['timestamp']}")
            print(f"Q: {exchange['question']}")
            print(f"R: {exchange['answer'][:200]}..." if len(exchange['answer']) > 200 else f"R: {exchange['answer']}")
        print("=" * 35)

    def clear_memory(self):
        """Clears conversation memory"""
        self.memory.clear_history()
        print("Conversation memory cleared.")

    def save_conversation(self, filename: str):
        """Saves the conversation"""
        self.memory.save_to_file(filename)
        print(f"Conversation saved to {filename}")

    def load_conversation(self, filename: str):
        """Loads a saved conversation"""
        self.memory.load_from_file(filename)
        print(f"Conversation loaded from {filename}")


def main():
    """Main function with improved user interface"""
    bot = AuroraBot()

    print("=== AuroraIq Chatbot ===")
    print("Special commands:")
    print("- 'quit' or 'exit': quit")
    print("- 'history': view history")
    print("- 'clear': clear memory")
    print("- 'save <filename>': save conversation")
    print("- 'load <filename>': load a conversation")
    print("- 'scoring': view hybrid scoring configuration")  # AJOUT: Nouvelle commande
    print("- 'weights <sem> <key> <ctx>': adjust weights (ex: weights 0.5 0.4 0.1)")  # AJOUT: Nouvelle commande
    print("=" * 50)  # MODIFICATION: Ligne plus longue

    while True:
        question = input("\n🤖 Your question: ").strip()

        if question.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        elif question.lower() == 'history':
            bot.show_history()
            continue

        elif question.lower() == 'clear':
            bot.clear_memory()
            continue

        # AJOUT: Nouvelle commande pour voir la configuration du scoring
        elif question.lower() == 'scoring':
            bot.show_scoring_info()
            continue

        # AJOUT: Nouvelle commande pour ajuster les poids
        elif question.lower().startswith('weights '):
            try:
                parts = question.split()
                if len(parts) == 4:
                    sem, key, ctx = float(parts[1]), float(parts[2]), float(parts[3])
                    bot.set_hybrid_weights(sem, key, ctx)
                else:
                    print("Format: weights <semantic> <keywords> <context>")
                    print("Example: weights 0.5 0.4 0.1")
            except ValueError:
                print("Error: please use decimal numbers")
            continue

        elif question.lower().startswith('save '):
            filename = question[5:].strip()
            if filename:
                bot.save_conversation(filename)
            else:
                print("Please specify a filename: save <filename>")
            continue

        elif question.lower().startswith('load '):
            filename = question[5:].strip()
            if filename:
                bot.load_conversation(filename)
            else:
                print("Please specify a filename: load <filename>")
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