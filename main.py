import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from langchain_community.embeddings import HuggingFaceEmbeddings
from flask import Flask, request, jsonify,session
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask import render_template, send_from_directory
import os
import sqlite3
import hashlib
import os
from datetime import datetime
from functools import wraps
import google.generativeai 
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
app.secret_key = 'your-super-secret-key-change-this-in-production'

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('laws_pdfs2', exist_ok=True)
DATABASE = 'users.db'

def get_db_connection():
    """Create and return database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with users table"""
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            organization TEXT,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, password_hash):
    """Verify password against hash"""
    return hashlib.sha256(password.encode()).hexdigest() == password_hash

def login_required(f):
    """Decorator to require login for protected routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

init_db()

@app.route('/')
def home():
    """Home page route"""
    return render_template('home.html')

@app.route('/register')
def register_page():
    """Registration page route"""
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register_submit():
    """Handle user registration"""
    try:
        first_name = request.form.get('firstName', '').strip()
        last_name = request.form.get('lastName', '').strip()
        email = request.form.get('email', '').strip().lower()
        organization = request.form.get('organization', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirmPassword', '')
        
        if not all([first_name, last_name, email, password]):
            flash('All required fields must be filled.', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('register.html')
        
        conn = get_db_connection()
        existing_user = conn.execute(
            'SELECT id FROM users WHERE email = ?', (email,)
        ).fetchone()
        
        if existing_user:
            conn.close()
            flash('An account with this email already exists.', 'error')
            return render_template('register.html')
        
        password_hash = hash_password(password)
        conn.execute('''
            INSERT INTO users (first_name, last_name, email, organization, password_hash)
            VALUES (?, ?, ?, ?, ?)
        ''', (first_name, last_name, email, organization or None, password_hash))
        
        conn.commit()
        conn.close()
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login_page'))
        
    except Exception as e:
        print(f"Registration error: {e}")
        flash('An error occurred during registration. Please try again.', 'error')
        return render_template('register.html')

@app.route('/login', methods=['GET'])
def login_page():
    """Login page route"""
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_submit():
    """Handle user login"""
    try:
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = request.form.get('remember')
        
        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password are required.'}), 400
        
        conn = get_db_connection()
        user = conn.execute('''
            SELECT id, first_name, last_name, email, password_hash
            FROM users WHERE email = ?
        ''', (email,)).fetchone()
        
        conn.close()
        
        if user and verify_password(password, user['password_hash']):
            session['user_id'] = user['id']
            session['user_name'] = f"{user['first_name']} {user['last_name']}"
            session['user_email'] = user['email']
            
            if remember:
                session.permanent = True
            
            return jsonify({'success': True, 'message': 'Login successful!'})
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password.'}), 401
            
    except Exception as e:
        return jsonify({'success': False, 'message': 'An error occurred during login.'}), 500

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard route - requires login"""
    return render_template('index.html')

@app.route('/logout')
def logout():
    """Logout route"""
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('home'))



class QueryType(Enum):
    CLAUSE_MEANING = "understanding_clause_meaning"
    DOCUMENT_SUMMARY = "summary_of_document"
    GENERAL_LAW_QUESTION = "general_question_from_law"
    USER_DOC_QUESTION = "question_from_doc_uploaded"

class LawTopic(Enum):
    administrative_and_goverance_rule = "administrative_and_goverance_rule"
    citizenship_and_immigration = "citizenship_and_immigration"
    criminal_and_penal_law = "criminal_and_penal_law"
    emergence_and_special_provisions = "emergence_and_special_provisions"
    enforcement_and_public_security = "enforcement_and_public_security"
    social_economic_and_cultural_and_political_acts = "social_economic_and_cultural_and_political_acts"

@dataclass
class QueryState:
    """State management for the LangGraph workflow"""
    query: str
    query_type: Optional[QueryType] = None
    law_topic: Optional[LawTopic] = None
    documents: Optional[List[Document]] = None
    answer: Optional[str] = None
    context: Optional[str] = None
    uploaded_file: Optional[str] = None
    messages: Annotated[List, add_messages] = None

class LegalAIAssistant:
    """Main class for legal AI assistant functionality"""
    
    def __init__(self, gemini_api_key: str):
        """Initialize the legal AI assistant with necessary components"""
        if not gemini_api_key:
            raise ValueError(" GEMINI_API_KEY is required")
        
        google.generativeai.configure(api_key=gemini_api_key)
        self.llm = GoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            api_key=gemini_api_key,
            temperature=0.3
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        self.clause_vectorstore = None
        self.laws_pdfs_vectorstore = {}
        self.user_doc_vectorstore = None
        
        self._load_clause_pdf()
        self._load_laws_pdfs()
        
        self.workflow = self._create_workflow()
    
    def _load_clause_pdf(self):
        """Load and index the clause.pdf document"""
        clause_path = "clause.pdf"
        if os.path.exists(clause_path):
            try:
                loader = PyPDFLoader(clause_path)
                documents = loader.load()
                chunks = self.text_splitter.split_documents(documents)
                self.clause_vectorstore = FAISS.from_documents(chunks, self.embeddings)
                logger.info("Clause PDF loaded and indexed successfully")
            except Exception as e:
                logger.error(f"Error loading clause PDF: {e}")
    
    def _load_laws_pdfs(self):
        """Load and index law books from the laws_pdfs directory"""
        laws_pdfs_dir = "laws_pdfs2"
        if not os.path.exists(laws_pdfs_dir):
            logger.warning(f"Law books directory {laws_pdfs_dir} not found")
            return
        
        for filename in os.listdir(laws_pdfs_dir):
            if filename.endswith('.pdf'):
                topic = filename.replace('.pdf', '').lower()
                try:
                    loader = PyPDFLoader(os.path.join(laws_pdfs_dir, filename))
                    documents = loader.load()
                    chunks = self.text_splitter.split_documents(documents)
                    self.laws_pdfs_vectorstore[topic] = FAISS.from_documents(chunks, self.embeddings)
                    logger.info(f"Law book {filename} loaded and indexed successfully")
                except Exception as e:
                    logger.error(f"Error loading law book {filename}: {e}")
    
    def _classify_query_intent(self, state: QueryState) -> QueryState:
        """Classify the user query into one of the predefined intents"""
        classification_prompt = """
        Classify the following query into one of these categories:
        1. understanding_clause_meaning - User wants to understand what is meaning of any difficult legal term or word
        2. summary_of_document - User wants a summary of a document
        3. general_question_from_law - User has a question related to clause or any section of law 
        4. question_from_doc_uploaded - User has a question about a document they uploaded
        
        Query: {query}
        
        Respond with only the category name.
        """
        
        try:
            result = self.llm.invoke(classification_prompt.format(query=state.query))
            query_type = QueryType(result.strip().lower())
            state.query_type = query_type
            logger.info(f"Query classified as: {query_type.value}")
        except Exception as e:
            logger.error(f"Error in query classification: {e}")
            state.query_type = QueryType.GENERAL_LAW_QUESTION
        
        return state
    
    def _classify_law_topic(self, state: QueryState) -> QueryState:
        """Classify the legal topic for general law questions"""
        if state.query_type != QueryType.GENERAL_LAW_QUESTION:
            return state
        
        topic_prompt = """
        Classify the following legal query into one of these topics:

        administrative_and_goverance_rule- This category deals with the laws, rules, and regulations that govern the operations and functions of government agencies, public bodies, and their officials.
        citizenship_and_immigration- This topic covers all legal matters related to the status of individuals as citizens or non-citizens.
        criminal_and_penal_law- This category is concerned with actions or omissions that are considered offenses against the state or society as a whole. 
        emergence_and_special_provisions- This category encompasses legal measures enacted to handle exceptional or crisis situations that fall outside of normal governance.
        enforcement_and_public_security- This topic focuses on the laws, policies, and actions of government agencies responsible for maintaining public order and safety.
        social_economic_and_cultural_and_political_acts- This broad category covers laws and policies that shape the social, economic, cultural, and political rights and well-being of a society.
        
        choose among these names only.

        Query: {query}
        
        Respond with only the topic name.
        """
        
        try:
            result = self.llm.invoke(topic_prompt.format(query=state.query))
            law_topic = LawTopic(result.strip().lower())
            state.law_topic = law_topic
            logger.info(f"Law topic classified as: {law_topic.value}")
        except Exception as e:
            logger.error(f"Error in law topic classification: {e}")
            state.law_topic = LawTopic.GENERAL
        
        return state
    
    def _handle_clause_meaning(self, state: QueryState) -> QueryState:
        """Handle clause meaning queries using clause.pdf or web search"""
        if not self.clause_vectorstore:
            return self._web_search_fallback(state)
        try:
            # Create retrieval chain with re-ranking
            compressor = LLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.clause_vectorstore.as_retriever(search_kwargs={"k": 5})
            )
            
            qa_prompt = PromptTemplate(
                template="""You are a legal expert. Explain the following clause or legal term in simple, accessible language.
                
                Context: {context}
                
                Question: {question}
                
                Provide a clear explanation that an average person can understand, including:
                1. What this clause means in plain English
                2. Why it's important
                3. Potential implications for the user
                
                Answer:""",
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=compression_retriever,
                chain_type_kwargs={"prompt": qa_prompt}
            )
            
            result = qa_chain.invoke({"query": state.query})
            state.answer = result["result"]
            
        except Exception as e:
            logger.error(f"Error in clause meaning handling: {e}")
            return self._web_search_fallback(state)
        
        return state
    
    def _handle_document_summary(self, state: QueryState) -> QueryState:
        """Handle document summary requests using uploaded document"""
        if not state.uploaded_file or not self.user_doc_vectorstore:
            state.answer = "Please upload a document first to get a summary."
            return state
        
        try:
            # Create retrieval chain for summarization
            retriever = self.user_doc_vectorstore.as_retriever(search_kwargs={"k": 10})
            
            summary_prompt = PromptTemplate(
                template="""You are a legal expert. Provide a comprehensive summary of the document based on the following content.
                
                Content: {context}
                
                Create a summary that includes:
                1. Document type and purpose
                2. Key parties involved
                3. Main terms and conditions
                4. Important dates and deadlines
                5. Rights and obligations
                6. Potential risks or concerns for the user
                
                Write in clear, simple language that anyone can understand.
                
                Summary:""",
                input_variables=["context"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": summary_prompt}
            )
            
            result = qa_chain.invoke({"query": "Provide a comprehensive summary of this document"})
            state.answer = result["result"]
            
        except Exception as e:
            logger.error(f"Error in document summary: {e}")
            state.answer = "Error generating document summary. Please try again."
        
        return state
    
    
    def _handle_general_law_question(self, state: QueryState) -> QueryState:
        """Handle general law questions using law books or web search"""
        if not state.law_topic:
            return self._web_search_fallback(state)
        
        vectorstore = self.laws_pdfs_vectorstore.get(state.law_topic.value)
        if not vectorstore:
            return self._web_search_fallback(state)
        
        try:

            compressor = LLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5}))
            
            law_prompt = PromptTemplate(
                template="""You are a highly precise and meticulous legal expert. Your goal is to provide the most accurate legal answer possible, based *exclusively* on the provided context.
                
                Context: {context}
                
                Question: {question}
                
                Task:
                1. Directly answer the question.
                2. Cite the exact relevant section or principle from the context.
                3. Explain the legal principle simply, without adding outside information.
                4. Do not guess or make assumptions. If the context does not contain the answer, state "The provided context does not contain the information to answer this question precisely."
                
                Answer:""",
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=compression_retriever,
                chain_type_kwargs={"prompt": law_prompt}
            )
            
            result = qa_chain.invoke({"query": state.query})
            state.answer = result["result"]
            
        except Exception as e:
            logger.error(f"Error in general law question handling: {e}")
            return self._web_search_fallback(state)
        
        return state
    
    def _handle_user_doc_question(self, state: QueryState) -> QueryState:
        """Enhanced RAG-based document question answering with deep content extraction"""
        if not self.user_doc_vectorstore:
            state.answer = "Please upload a document first to ask questions about it."
            return state
        
        try:
            from langchain.retrievers import (
                ContextualCompressionRetriever, 
                EnsembleRetriever,
                MultiQueryRetriever
            )
            from langchain.retrievers.document_compressors import (
                LLMChainExtractor,
                EmbeddingsFilter,
                DocumentCompressorPipeline
            )
            from langchain.chains import RetrievalQA
            from langchain.prompts import PromptTemplate
            
            
            dense_retriever = self.user_doc_vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8, "fetch_k": 16, "lambda_mult": 0.7}
            )
            sparse_retriever = self.user_doc_vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            ensemble_retriever = EnsembleRetriever(
                retrievers=[dense_retriever, sparse_retriever],
                weights=[0.7, 0.3]
            )
            
            
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=ensemble_retriever,
                llm=self.llm,
                prompt=PromptTemplate(
                    template="""Generate 3 different search queries for: {question}
                    
                    Alternative queries:""",
                    input_variables=["question"]
                )
            )
            
            
            compressors = []
            
            
            try:
                embeddings_filter = EmbeddingsFilter(
                    embeddings=self.embeddings_model,
                    similarity_threshold=0.6,
                    k=15
                )
                compressors.append(embeddings_filter)
            except:
                pass
                
            
            llm_extractor = LLMChainExtractor.from_llm(
                llm=self.llm,
                prompt=PromptTemplate(
                    template="""Extract relevant information for: {question}
                    
                    Text: {context}
                    
                    Relevant information:""",
                    input_variables=["question", "context"]
                )
            )
            compressors.append(llm_extractor)
            
            compressor = DocumentCompressorPipeline(compressors=compressors) if len(compressors) > 1 else llm_extractor
            
            
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=multi_query_retriever
            )
            
            
            enhanced_prompt = PromptTemplate(
                template="""You are an expert document analyst. Answer comprehensively using the context.

                            Context: {context}

                            Question: {question}

                            Provide a detailed answer that:
                            1. Directly answers the question
                            2. Quotes specific relevant passages
                            3. Explains implications and significance
                            4. Uses clear, accessible language
                            5. States if information is incomplete

                            Answer:""",
                input_variables=["context", "question"]
            )
            
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=compression_retriever,
                chain_type_kwargs={"prompt": enhanced_prompt},
                return_source_documents=True
            )
            
            
            result = qa_chain.invoke({"query": state.query})
            answer = result["result"].strip()
            
            
            if result.get("source_documents"):
                answer += f"\n\n Based on {len(result['source_documents'])} document sections."
            
            
            uncertainty_indicators = ["don't know", "not clear", "cannot determine", "insufficient"]
            if any(indicator in answer.lower() for indicator in uncertainty_indicators):
                answer = f"⚠️ **Partial Information**: {answer}"
                
            state.answer = answer
            
        except Exception as e:
            logger.error(f"Enhanced RAG failed: {e}")
            
            try:
                compressor = LLMChainExtractor.from_llm(self.llm)
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.user_doc_vectorstore.as_retriever(search_kwargs={"k": 8})
                )
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=compression_retriever,
                    chain_type_kwargs={
                        "prompt": PromptTemplate(
                            template="""Answer based on context: {context}
                            
                            Question: {question}
                            
                            Answer:""",
                            input_variables=["context", "question"]
                        )
                    }
                )
                
                result = qa_chain.invoke({"query": state.query})
                state.answer = result["result"]
                
            except Exception as fallback_error:
                logger.error(f"Fallback failed: {fallback_error}")
                state.answer = "Error processing document question. Please try again."
        
        return state


    def _web_search_fallback(self, state: QueryState) -> QueryState:
        """Fallback to web search when local documents don't have the answer"""
        try:
            search_query = f"legal {state.query}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            search_url = f"https://www.google.com/search?q={search_query}"
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                search_results = []
                
                for result in soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd')[:3]:
                    search_results.append(result.get_text())
                
                if search_results:
                    context = " ".join(search_results)
                    web_answer_prompt = f"""
                    Based on the following web search results, provide a clear answer to the legal question:
                    
                    Question: {state.query}
                    
                    Search Results: {context}
                    
                    Provide a helpful answer in simple language, but note that this is general information and not legal advice.
                    """
                    
                    result = self.llm.invoke(web_answer_prompt)
                    state.answer = f"{result}\n\n*Note: This information is based on web search and is for general guidance only. Please consult a legal professional for specific advice.*"
                else:
                    state.answer = "I couldn't find specific information about your query. Please consult a legal professional for advice."
            else:
                state.answer = "I couldn't search for additional information at the moment. Please consult a legal professional for advice."
                
        except Exception as e:
            logger.error(f"Error in web search fallback: {e}")
            state.answer = "I couldn't find specific information about your query. Please consult a legal professional for advice."
        
        return state
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for query processing"""
        workflow = StateGraph(QueryState)
        
        # Add nodes
        workflow.add_node("classify_intent", self._classify_query_intent)
        workflow.add_node("classify_topic", self._classify_law_topic)
        workflow.add_node("handle_clause", self._handle_clause_meaning)
        workflow.add_node("handle_summary", self._handle_document_summary)
        workflow.add_node("handle_general", self._handle_general_law_question)
        workflow.add_node("handle_user_doc", self._handle_user_doc_question)
        
        workflow.set_entry_point("classify_intent")
        
        def route_by_intent(state: QueryState) -> str:
            if state.query_type == QueryType.CLAUSE_MEANING:
                return "handle_clause"
            elif state.query_type == QueryType.DOCUMENT_SUMMARY:
                return "handle_summary"
            elif state.query_type == QueryType.GENERAL_LAW_QUESTION:
                return "classify_topic"
            elif state.query_type == QueryType.USER_DOC_QUESTION:
                return "handle_user_doc"
            else:
                return "handle_general"
        
        def route_after_topic(state: QueryState) -> str:
            return "handle_general"
        
        
        workflow.add_conditional_edges(
            "classify_intent",
            route_by_intent,
            {
                "handle_clause": "handle_clause",
                "handle_summary": "handle_summary",
                "classify_topic": "classify_topic",
                "handle_user_doc": "handle_user_doc",
                "handle_general": "handle_general"
            }
        )
        
        workflow.add_conditional_edges(
            "classify_topic",
            route_after_topic,
            {"handle_general": "handle_general"}
        )
        
        
        workflow.add_edge("handle_clause", END)
        workflow.add_edge("handle_summary", END)
        workflow.add_edge("handle_general", END)
        workflow.add_edge("handle_user_doc", END)
        
        return workflow.compile()
    
    def process_document(self, file_path: str) -> bool:
        """Process and index an uploaded document"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            self.user_doc_vectorstore = FAISS.from_documents(chunks, self.embeddings)
            logger.info(f"Document {file_path} processed and indexed successfully")
            return True
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return False
    
    def query(self, user_query: str, uploaded_file: str = None) -> Dict[str, Any]:
        """Main query processing method"""
        try:
            state = QueryState(query=user_query, uploaded_file=uploaded_file)
            result = self.workflow.invoke(state)

            query_type = result.get("query_type")
            law_topic = result.get("law_topic")
            answer = result.get("answer")
            
            return {
            "query": user_query,
            "query_type": getattr(query_type, "value", query_type),
            "law_topic": getattr(law_topic, "value", law_topic),
            "answer": answer,
            "success": True
        }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": user_query,
                "answer": "An error occurred while processing your query. Please try again.",
                "success": False,
                "error": str(e)
            }


legal_assistant = None

@app.route('/upload', methods=['POST'])
def upload_document():
    """Endpoint for uploading documents"""
    global legal_assistant
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        
        success = legal_assistant.process_document(file_path)
        
        if success:
            return jsonify({
                "message": "Document uploaded and processed successfully",
                "filename": filename,
                "success": True
            })
        else:
            return jsonify({"error": "Failed to process document"}), 500
    
    return jsonify({"error": "Only PDF files are supported"}), 400

@app.route('/query', methods=['POST'])
def handle_query():
    """Main endpoint for handling user queries"""
    global legal_assistant
    
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        
        user_query = data['query']
        uploaded_file = data.get('uploaded_file')
        
        result = legal_assistant.query(user_query, uploaded_file)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in query endpoint: {e}")
        return jsonify({
            "error": "Internal server error",
            "success": False
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "Legal AI Assistant"})

def create_app(gemini_api_key: str):
    """Create and configure the Flask application"""
    global legal_assistant
    
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('laws_pdfs2', exist_ok=True)

    try:
        legal_assistant = LegalAIAssistant(gemini_api_key)
        logger.info("Legal AI Assistant initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Legal AI Assistant: {e}")
        raise
    
    return app



if __name__ == '__main__':
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    app = create_app(gemini_api_key)
    app.run(debug=True, host='0.0.0.0', port=5000)
