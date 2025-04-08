# import os
# import chromadb
# from groq import Groq
# from typing import List, Dict

# class ContextAwareChatbot:
#     def __init__(
#         self, 
#         groq_api_key: str, 
#         chroma_db_path: str = './chroma_storage',
#         collection_name: str = 'document_collection',
#         max_context_tokens: int = 1000
#     ):
#         """
#         Initialize the chatbot with Groq API and ChromaDB configuration
        
#         Args:
#             groq_api_key (str): Groq API key for language model
#             chroma_db_path (str): Path to ChromaDB persistent storage
#             collection_name (str): Name of the ChromaDB collection
#             max_context_tokens (int): Maximum tokens for context retrieval
#         """
#         # Initialize Groq client
#         self.groq_client = Groq(api_key=groq_api_key)
        
#         # Initialize ChromaDB client
#         self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
#         self.collection = self.chroma_client.get_collection(name=collection_name)
        
#         # Conversation history management
#         self.conversation_history: List[Dict[str, str]] = []
#         self.max_context_tokens = max_context_tokens
        
#         # Embedding model for query processing
#         from sentence_transformers import SentenceTransformer
#         self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

#     def _retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
#         """
#         Retrieve relevant context from ChromaDB based on query
        
#         Args:
#             query (str): User's query
#             top_k (int): Number of top relevant contexts to retrieve
        
#         Returns:
#             List of relevant context passages
#         """
#         # Generate embedding for the query
#         query_embedding = self.embedding_model.encode([query])[0].tolist()
        
#         # Retrieve similar passages from ChromaDB
#         results = self.collection.query(
#             query_embeddings=[query_embedding],
#             n_results=top_k
#         )
        
#         return results['documents'][0] if results['documents'] else []

#     def _manage_conversation_history(self, new_message: Dict[str, str]) -> List[Dict[str, str]]:
#         """
#         Manage conversation history to stay within token limit
        
#         Args:
#             new_message (dict): New message to add to history
        
#         Returns:
#             Trimmed conversation history
#         """
#         self.conversation_history.append(new_message)
        
#         # Simple token estimation and trimming
#         while len(str(self.conversation_history)) > self.max_context_tokens:
#             self.conversation_history.pop(0)
        
#         return self.conversation_history

#     def chat(self, user_query: str) -> str:
#         """
#         Main chat method to process user query and generate response
        
#         Args:
#             user_query (str): User's input message
        
#         Returns:
#             AI-generated response
#         """
#         # Retrieve relevant context
#         relevant_contexts = self._retrieve_relevant_context(user_query)
        
#         # Prepare context-augmented prompt
#         context_str = "\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(relevant_contexts)])
        
#         # Prepare conversation history
#         history_str = "\n".join([
#             f"{msg['role'].capitalize()}: {msg['content']}" 
#             for msg in self._manage_conversation_history({'role': 'user', 'content': user_query})
#         ])
        
#         # Construct full prompt
#         full_prompt = f"""
#         Relevant Document Contexts:
#         {context_str}
        
#         Conversation History:
#         {history_str}
        
#         Please provide a helpful and concise response to the latest user query, 
#         leveraging the context from retrieved documents and conversation history.
#         """
        
#         # Generate response using Groq
#         try:
#             chat_completion = self.groq_client.chat.completions.create(
#                 messages=[
#                     {"role": "system", "content": "You are a helpful AI assistant."},
#                     {"role": "user", "content": full_prompt}
#                 ],
#                 model="llama3-8b-8192"  # You can change the model as needed
#             )
            
#             response = chat_completion.choices[0].message.content
            
#             # Add AI response to conversation history
#             self._manage_conversation_history({'role': 'assistant', 'content': response})
            
#             return response
        
#         except Exception as e:
#             return f"An error occurred: {str(e)}"

# # Example Usage
# def main():
     
#     # Replace with your actual Groq API key
#     GROQ_API_KEY = "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"
    
#     try:
#         # Initialize chatbot
#         chatbot = ContextAwareChatbot(
#             groq_api_key=GROQ_API_KEY,
#             chroma_db_path= './chroma_storage',
#             collection_name= 'technical_docs'
#         )
        
#         # Interactive chat loop
#         while True:
#             user_input = input("You: ")
#             if user_input.lower() in ['exit', 'quit', 'bye']:
#                 break
            
#             response = chatbot.chat(user_input)
#             print("AI:", response)
    
#     except Exception as e:
#         print(f"Chatbot initialization error: {e}")

# # Uncomment to run
# if __name__ == "__main__":
#     main()





























# import streamlit as st
# import os
# import chromadb
# from groq import Groq
# from typing import List, Dict
# from sentence_transformers import SentenceTransformer

# class ContextAwareChatbot:
#     def __init__(
#         self, 
#         groq_api_key: str, 
#         chroma_db_path: str = './chroma_storage',
#         collection_name: str = 'document_collection',
#         max_context_tokens: int = 1000
#     ):
#         """
#         Initialize the chatbot with Groq API and ChromaDB configuration
        
#         Args:
#             groq_api_key (str): Groq API key for language model
#             chroma_db_path (str): Path to ChromaDB persistent storage
#             collection_name (str): Name of the ChromaDB collection
#             max_context_tokens (int): Maximum tokens for context retrieval
#         """
#         # Initialize Groq client
#         self.groq_client = Groq(api_key=groq_api_key)
        
#         # Initialize ChromaDB client
#         self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
#         self.collection = self.chroma_client.get_collection(name=collection_name)
        
#         # Conversation history management
#         self.conversation_history: List[Dict[str, str]] = []
#         self.max_context_tokens = max_context_tokens
        
#         # Embedding model for query processing
#         self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

#     def _retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
#         """
#         Retrieve relevant context from ChromaDB based on query
        
#         Args:
#             query (str): User's query
#             top_k (int): Number of top relevant contexts to retrieve
        
#         Returns:
#             List of relevant context passages
#         """
#         # Generate embedding for the query
#         query_embedding = self.embedding_model.encode([query])[0].tolist()
        
#         # Retrieve similar passages from ChromaDB
#         results = self.collection.query(
#             query_embeddings=[query_embedding],
#             n_results=top_k
#         )
        
#         return results['documents'][0] if results['documents'] else []

#     def _manage_conversation_history(self, new_message: Dict[str, str]) -> List[Dict[str, str]]:
#         """
#         Manage conversation history to stay within token limit
        
#         Args:
#             new_message (dict): New message to add to history
        
#         Returns:
#             Trimmed conversation history
#         """
#         self.conversation_history.append(new_message)
        
#         # Simple token estimation and trimming
#         while len(str(self.conversation_history)) > self.max_context_tokens:
#             self.conversation_history.pop(0)
        
#         return self.conversation_history

#     def chat(self, user_query: str) -> str:
#         """
#         Main chat method to process user query and generate response
        
#         Args:
#             user_query (str): User's input message
        
#         Returns:
#             AI-generated response
#         """
#         # Retrieve relevant context
#         relevant_contexts = self._retrieve_relevant_context(user_query)
        
#         # Prepare context-augmented prompt
#         context_str = "\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(relevant_contexts)])
        
#         # Prepare conversation history
#         history_str = "\n".join([
#             f"{msg['role'].capitalize()}: {msg['content']}" 
#             for msg in self._manage_conversation_history({'role': 'user', 'content': user_query})
#         ])
        
#         # Construct full prompt
#         full_prompt = f"""
#         Relevant Document Contexts:
#         {context_str}
        
#         Conversation History:
#         {history_str}
        
#         Please provide a helpful and concise response to the latest user query, 
#         leveraging the context from retrieved documents and conversation history.
#         """
        
#         # Generate response using Groq
#         try:
#             chat_completion = self.groq_client.chat.completions.create(
#                 messages=[
#                     {"role": "system", "content": "You are a helpful AI assistant."},
#                     {"role": "user", "content": full_prompt}
#                 ],
#                 model="llama3-8b-8192"  # You can change the model as needed
#             )
            
#             response = chat_completion.choices[0].message.content
            
#             # Add AI response to conversation history
#             self._manage_conversation_history({'role': 'assistant', 'content': response})
            
#             return response
        
#         except Exception as e:
#             return f"An error occurred: {str(e)}"

# def main():
#     # Streamlit UI setup
#     st.set_page_config(page_title="Context-Aware Chatbot", page_icon="💬")
    
#     # Title and description
#     st.title("🤖 Context-Aware Document Chatbot")
#     st.write("Chat with your documents using advanced context retrieval")

#     # Groq API Key input
#     GROQ_API_KEY = st.sidebar.text_input("Enter Groq API Key", type="password")
    
#     # Chatbot initialization
#     if GROQ_API_KEY:
#         try:
#             # Initialize chatbot
#             chatbot = ContextAwareChatbot(
#                 groq_api_key=GROQ_API_KEY,
#                 chroma_db_path='./chroma_storage',
#                 collection_name='technical_docs'
#             )
            
#             # Initialize chat history in session state
#             if 'messages' not in st.session_state:
#                 st.session_state.messages = []
            
#             # Display chat messages from history on app rerun
#             for message in st.session_state.messages:
#                 with st.chat_message(message["role"]):
#                     st.markdown(message["content"])
            
#             # Chat input
#             if prompt := st.chat_input("What would you like to know?"):
#                 # Add user message to chat history
#                 st.session_state.messages.append({"role": "user", "content": prompt})
                
#                 # Display user message
#                 with st.chat_message("user"):
#                     st.markdown(prompt)
                
#                 # Generate and display assistant response
#                 with st.chat_message("assistant"):
#                     response = chatbot.chat(prompt)
#                     st.markdown(response)
                
#                 # Add assistant response to chat history
#                 st.session_state.messages.append({"role": "assistant", "content": response})
        
#         except Exception as e:
#             st.error(f"Chatbot initialization error: {e}")
#     else:
#         st.warning("Please enter your Groq API Key in the sidebar")

# if __name__ == "__main__":
#     main()








__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from groq import Groq
from typing import List

class Chatbot:
    def __init__(
        self,
        groq_api_key: str = "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",  # Directly inserted key
        chroma_db_path: str = './chroma_storage',
        collection_name: str = 'document_collection',
        max_context_tokens: int = 1000
    ):
        """
        Initialize the chatbot with Groq API and ChromaDB configuration
        """
        self.client = Groq(api_key=groq_api_key)
        self.chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=chroma_db_path))

        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

        self.max_context_tokens = max_context_tokens

    def add_document(self, doc_id: str, content: str, metadata: dict = {}):
        """
        Add a document to the ChromaDB collection
        """
        self.collection.add(documents=[content], ids=[doc_id], metadatas=[metadata])

    def query_documents(self, query: str, n_results: int = 5):
        """
        Retrieve top documents related to the query from ChromaDB
        """
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return results.get("documents", [[]])[0]

    def construct_prompt(self, context_docs: List[str], question: str):
        """
        Construct a prompt using retrieved documents and the user's question
        """
        context = "\n\n".join(context_docs)
        prompt = f"""You are a helpful assistant. Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:"""
        return prompt

    def ask(self, question: str, model: str = "mixtral-8x7b-32768", n_results: int = 5) -> str:
        """
        Ask a question to the chatbot. It retrieves documents, constructs a prompt, and queries the model.
        """
        context_docs = self.query_documents(question, n_results=n_results)
        prompt = self.construct_prompt(context_docs, question)

        chat_completion = self.client.chat.completions.create(
            model=model,
            messages=[ 
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )

        return chat_completion.choices[0].message.content

    def persist(self):
        """
        Persist ChromaDB state to disk
        """
        self.chroma_client.persist()
