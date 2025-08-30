# !pip install -q gradio groq faiss-cpu langchain pypdf sentence-transformers langchain_community langchain-huggingface PyPDF2 openai
# from groq import Groq

import os
import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from google.colab import userdata
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# URL of your Azure Key Vault
KEYVAULT_URL = "https://test-key-vaultt.vault.azure.net/"

# Authenticate using DefaultAzureCredential
credential = DefaultAzureCredential()
kv_client = SecretClient(vault_url=KEYVAULT_URL, credential=credential)

# Fetch secrets
subscription_key = kv_client.get_secret("subscription-key").value
api_version = kv_client.get_secret("api-version").value
azure_endpoint = kv_client.get_secret("azure-endpoint").value


client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=azure_endpoint,
    api_key=subscription_key,
)

MODEL="gpt-5-chat"

# 📌 Path to store FAISS index (persistent DB)
DB_PATH = "faiss_index"

# Global DB
db = None

# Load existing DB if present
if os.path.exists(DB_PATH):
    db = FAISS.load_local(
        DB_PATH,
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
    )

# 0️⃣ Function to clear DB
def clear_db():
    global db
    db = None
    if os.path.exists(DB_PATH):
        import shutil
        shutil.rmtree(DB_PATH)   # remove FAISS index folder completely
    return None, "🧹 Knowledge base cleared. Please upload SOPs again."
    
# 1️⃣ Process uploaded PDFs
def process_pdfs(pdf_files):
    global db
    if not pdf_files:
        return None, "⚠️ Please upload at least one PDF."

    all_texts = []
    for pdf in pdf_files:
        reader = PdfReader(pdf.name)
        text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
        all_texts.append(text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents(all_texts)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if db is None:
        db_new = FAISS.from_documents(docs, embeddings)
        db_new.save_local(DB_PATH)
        db = db_new
    else:
        db.add_documents(docs)
        db.save_local(DB_PATH)

    return db, f"✅ {len(pdf_files)} SOP(s) indexed. Total chunks: {len(docs)}"


# 2️⃣ Chat function (retrieval + Azure LLaMA)
def chat_with_sop(question: str):
    if db is None:
        return "⚠️ No SOP database found. Please upload PDFs first."

    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)   # ✅ use invoke instead of deprecated get_relevant_documents
    context = "\n\n".join([d.page_content for d in docs])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an assistant that answers questions based only on the provided SOPs."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=1500,
        temperature=0.7,
        top_p=1.0,
    )

    return response.choices[0].message.content


# 3️⃣ Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        gr.Markdown(
            """
            # 📘 SOP AI Assistant
            Upload your **Mainframe / AutoSys SOPs** 📂 and ask intelligent questions 🤖.
            The assistant retrieves context from your uploaded PDFs and gives **precise answers**.
            """
        )

    with gr.Tab("📂 Upload SOPs"):
        with gr.Row():
            file_input = gr.File(
                label="Upload SOP PDFs",
                type="filepath",
                file_types=[".pdf"],
                file_count="multiple"
            )
        status = gr.Textbox(
            label="📊 Upload Status",
            interactive=False,
            placeholder="Upload SOP documents to build knowledge base...",
            lines=2
        )
        output = gr.State()
        file_input.upload(process_pdfs, inputs=file_input, outputs=[output, status])

    with gr.Tab("💬 Chat with SOPs"):
        with gr.Row():
            question = gr.Textbox(
                label="❓ Ask a Question",
                placeholder="e.g., What are the restart steps for job XYZ123?",
                lines=2
            )
        with gr.Row():
            answer = gr.Textbox(
                label="🤖 Assistant Answer",
                interactive=False,
                lines=8,
                show_copy_button=True
            )
        with gr.Row():
            clear_btn = gr.Button("🧹 Clear Chat")

        question.submit(chat_with_sop, inputs=question, outputs=answer)
        clear_btn.click(lambda: ("", ""), None, [question, answer])

    gr.Markdown(
        """
        ---
        🔒 *Your documents stay local. Only small text chunks are sent to Azure for answers.*
        💡 Powered by **LangChain + FAISS + Azure LLaMA**.
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8000)))