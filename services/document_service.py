"""Document processing and indexing service."""
import os
import logging
from typing import Optional, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import Config

logger = logging.getLogger(__name__)


class DocumentService:
    """Handles document processing and vector storage."""
    
    def __init__(self):
        """Initialize document service."""
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len
        )
        
        self.clause_vectorstore: Optional[FAISS] = None
        self.laws_pdfs_vectorstore: Dict[str, FAISS] = {}
        self.user_doc_vectorstore: Optional[FAISS] = None
    
    def load_clause_pdf(self, clause_path: str = "clause.pdf") -> bool:
        """Load and index the clause.pdf document."""
        if not os.path.exists(clause_path):
            logger.warning(f"Clause PDF not found at {clause_path}")
            return False
        
        try:
            loader = PyPDFLoader(clause_path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            self.clause_vectorstore = FAISS.from_documents(chunks, self.embeddings)
            logger.info("Clause PDF loaded and indexed successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading clause PDF: {e}")
            return False
    
    def load_law_pdfs(self, laws_dir: str = None) -> None:
        """Load and index law books from directory."""
        laws_dir = laws_dir or Config.LAWS_PDF_FOLDER
        
        if not os.path.exists(laws_dir):
            logger.warning(f"Law books directory {laws_dir} not found")
            return
        
        for filename in os.listdir(laws_dir):
            if filename.endswith('.pdf'):
                topic = filename.replace('.pdf', '').lower()
                try:
                    loader = PyPDFLoader(os.path.join(laws_dir, filename))
                    documents = loader.load()
                    chunks = self.text_splitter.split_documents(documents)
                    self.laws_pdfs_vectorstore[topic] = FAISS.from_documents(
                        chunks, self.embeddings
                    )
                    logger.info(f"Law book {filename} loaded and indexed")
                except Exception as e:
                    logger.error(f"Error loading law book {filename}: {e}")
    
    def process_user_document(self, file_path: str) -> bool:
        """Process and index an uploaded document."""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            self.user_doc_vectorstore = FAISS.from_documents(chunks, self.embeddings)
            logger.info(f"Document {file_path} processed and indexed")
            return True
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return False