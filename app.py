import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible for the provided context. 
    If the answer is not in the provided context, just say "Answer is not available in the context".
    Context:\n {context}?\n
    Question:\n {question}?\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[-1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = "".join([item["text"] for item in transcript_text])
        return transcript
    except Exception as e:
        raise Exception("Error extracting transcript: " + str(e))

def generate_gemini_summary(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        if not docs:
            st.warning("No relevant information found in the uploaded PDFs for the given question.")
            return
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"An error occurred: {e}")

def main():
    st.set_page_config(page_title="Document & Video Summarizer", layout="wide", page_icon="📄")
    st.title("Document and Video Summarizer 🧠")

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Missing Google API Key. Please set it in the .env file.")
        return

    option = st.selectbox("Choose a task:", ["Chat with PDFs", "Summarize YouTube Video"])

    if option == "Chat with PDFs":
        st.subheader("Chat with Multiple PDF Files")
        user_question = st.text_input("Ask a Question from the PDF files...")

        if user_question:
            user_input(user_question)

        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files and Click Submit", accept_multiple_files=True)

            if st.button("Submit & Process"):
                if pdf_docs:
                    with st.spinner("Processing..."):
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.warning("Uploaded PDFs contain no text. Please upload valid PDFs.")
                            return
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("PDFs processed successfully! You can now ask questions.")
                else:
                    st.warning("Please upload at least one PDF file.")

    elif option == "Summarize YouTube Video":
        st.subheader("Summarize YouTube Video")
        youtube_video_url = st.text_input("Enter YouTube Video URL:")

        if youtube_video_url:
            try:
                with st.spinner("Extracting transcript..."):
                    transcript_text = extract_transcript_details(youtube_video_url)

                y_prompt = """
                You are a YouTube video summarizer. You will summarize the transcript text 
                and provide the key points within 250 words. Here is the transcript:
                """

                with st.spinner("Generating summary..."):
                    summary = generate_gemini_summary(transcript_text, y_prompt)

                st.subheader("Video Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()