"""Query handlers for different query types."""
import logging
import requests
from bs4 import BeautifulSoup
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever, MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter, DocumentCompressorPipeline
from langchain_google_genai import GoogleGenerativeAI
from models.enums import QueryState
from services.document_service import DocumentService

logger = logging.getLogger(__name__)


class QueryHandlers:
    """Handles different types of queries."""
    
    def __init__(self, llm: GoogleGenerativeAI, doc_service: DocumentService):
        """Initialize handlers."""
        self.llm = llm
        self.doc_service = doc_service
    
    def handle_clause_meaning(self, state: QueryState) -> QueryState:
        """Handle clause meaning queries."""
        if not self.doc_service.clause_vectorstore:
            return self._web_search_fallback(state)
        
        try:
            compressor = LLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.doc_service.clause_vectorstore.as_retriever(
                    search_kwargs={"k": 5}
                )
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
    
    def handle_document_summary(self, state: QueryState) -> QueryState:
        """Handle document summary requests."""
        if not state.uploaded_file or not self.doc_service.user_doc_vectorstore:
            state.answer = "Please upload a document first to get a summary."
            return state
        
        try:
            retriever = self.doc_service.user_doc_vectorstore.as_retriever(
                search_kwargs={"k": 10}
            )
            
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
    
    def handle_general_law_question(self, state: QueryState) -> QueryState:
        """Handle general law questions."""
        if not state.law_topic:
            return self._web_search_fallback(state)
        
        vectorstore = self.doc_service.laws_pdfs_vectorstore.get(state.law_topic.value)
        if not vectorstore:
            return self._web_search_fallback(state)
        
        try:
            compressor = LLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
            )
            
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
    
    def handle_user_doc_question(self, state: QueryState) -> QueryState:
        """Handle questions about uploaded documents."""
        if not self.doc_service.user_doc_vectorstore:
            state.answer = "Please upload a document first to ask questions about it."
            return state
        
        try:
            # Dense retriever (MMR)
            dense_retriever = self.doc_service.user_doc_vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8, "fetch_k": 16, "lambda_mult": 0.7}
            )
            
            # Sparse retriever
            sparse_retriever = self.doc_service.user_doc_vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            # Combined retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[dense_retriever, sparse_retriever],
                weights=[0.7, 0.3]
            )
            
            # Multi-query generation
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=ensemble_retriever,
                llm=self.llm,
                prompt=PromptTemplate(
                    template="Generate 3 alternative search queries for: {question}\n\nAlternatives:",
                    input_variables=["question"]
                )
            )
            
            # Build compressor pipeline
            compressors = []
            
            try:
                embeddings_filter = EmbeddingsFilter(
                    embeddings=self.doc_service.embeddings,
                    similarity_threshold=0.60,
                    k=15
                )
                compressors.append(embeddings_filter)
            except Exception as e:
                logger.warning(f"Embeddings filter unavailable: {e}")
            
            llm_extractor = LLMChainExtractor.from_llm(self.llm)
            compressors.append(llm_extractor)
            
            compressor = (DocumentCompressorPipeline(compressors=compressors) 
                         if len(compressors) > 1 else compressors[0])
            
            compression_retriever = ContextualCompressionRetriever(
                base_retriever=multi_query_retriever,
                base_compressor=compressor
            )
            
            qa_prompt = PromptTemplate(
                template="""You are a legal analysis expert. Based strictly on the provided document context, answer the user's question.

                Context: {context}
                Question: {question}

                Instructions:
                1. Answer using ONLY the information from the context.
                2. Cite the relevant part of the document when possible.
                3. If the document does not contain the answer, respond:
                   "The uploaded document does not contain information relevant to this question."
                4. Do not hallucinate or infer anything outside the given text.

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
            logger.error(f"Error in user document QA: {e}")
            state.answer = "An error occurred while processing your question. Please try again."
        
        return state
    
    def _web_search_fallback(self, state: QueryState) -> QueryState:
        """Fallback to web search when local documents don't have the answer."""
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