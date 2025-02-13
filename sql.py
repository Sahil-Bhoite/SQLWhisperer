import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
import logging

# Configuration
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
SYSTEM_PROMPT = """You are an expert SQL engineer. Follow these rules:
1. Use proper JOIN syntax based on schema relationships
2. Include necessary WHERE clauses
3. Use appropriate aggregate functions
4. Follow ANSI SQL standards
5. Include clear explanations in Markdown"""

class UniversalQueryGPT:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.3,
            model_name="deepseek-r1-distill-llama-70b",
            groq_api_key=GROQ_API_KEY
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=256,
            separators=['\n\n', '\n', ';', ' ', '']
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
    def process_files(self, files):
        """Process uploaded files into vector stores"""
        schema_docs = []
        query_docs = []
        
        for file in files:
            content = self.extract_text(file)
            if "CREATE TABLE" in content.upper():
                chunks = self.text_splitter.split_text(content)
                schema_docs.extend([Document(page_content=c) for c in chunks])
            else:
                chunks = self.text_splitter.split_text(content)
                query_docs.extend([Document(page_content=c) for c in chunks])
        
        return {
            "schema_store": FAISS.from_documents(schema_docs, self.embeddings) if schema_docs else None,
            "query_store": FAISS.from_documents(query_docs, self.embeddings) if query_docs else None
        }

    def extract_text(self, file):
        try:
            return file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            return str(file.getvalue())

    def generate_sql(self, context, query):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", 
             "Schema Context:\n{schemas}\n\n"
             "Example Queries:\n{samples}\n\n"
             "User Question: {query}\n\n"
             "Generate SQL and explanation:")
        ])
        
        chain = prompt_template | self.llm | StrOutputParser()
        return chain.invoke({
            "schemas": context.get("schemas", ""),
            "samples": context.get("samples", ""),
            "query": query
        })

def main():
    st.set_page_config(page_title="SQLWhisperer a Universal QueryGPT", layout="wide")
    st.header("Universal Natural Language to SQL Converter")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_stores" not in st.session_state:
        st.session_state.vector_stores = None

    # Sidebar for file upload
    with st.sidebar:
        st.title("Database Setup")
        uploaded_files = st.file_uploader(
            "Upload SQL schemas and sample queries (SQL/txt files)",
            accept_multiple_files=True,
            type=["sql", "txt"]
        )
        
        if st.button("Process Schema"):
            if uploaded_files:
                with st.spinner("Analyzing database structure..."):
                    qgpt = UniversalQueryGPT()
                    st.session_state.vector_stores = qgpt.process_files(uploaded_files)
                    st.success("Database schema processed!")
            else:
                st.warning("Please upload schema files first")

    # Main chat interface
    user_query = st.chat_input("Ask a data question in plain English...")
    
    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        with st.spinner("Generating SQL query..."):
            try:
                qgpt = UniversalQueryGPT()
                context = {}
                
                if st.session_state.vector_stores:
                    # Retrieve relevant schema and query context
                    schema_docs = st.session_state.vector_stores["schema_store"].similarity_search(
                        user_query, k=5)
                    query_docs = st.session_state.vector_stores["query_store"].similarity_search(
                        user_query, k=3)
                    
                    context = {
                        "schemas": "\n".join([d.page_content for d in schema_docs]),
                        "samples": "\n".join([d.page_content for d in query_docs])
                    }
                
                response = qgpt.generate_sql(context, user_query)
                st.session_state.chat_history.append(AIMessage(content=response))
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.session_state.chat_history.append(AIMessage(
                    content="Sorry, I encountered an error processing your request."
                ))

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                try:
                    sql_part = message.content.split("```sql")[1].split("```")[0]
                    explanation_part = message.content.split("Explanation:")[1]
                    st.code(sql_part, language="sql")
                    st.markdown(f"**Explanation:** {explanation_part}")
                except IndexError:
                    st.markdown(message.content)

if __name__ == "__main__":
    main()