import os
import streamlit as st
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders.youtube import YoutubeLoader
import re
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from youtube_transcript_api._errors import NoTranscriptFound
from pytube import YouTube
import assemblyai as aai

# Load environment variables
load_dotenv()

# Set up Streamlit app
st.title("📽️ Youtube video chatbot 🤖")
st.sidebar.title("YouTube URL")
url = st.sidebar.text_input("Enter YouTube URL")

# Display the entered URL
process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()
llm = OpenAI(temperature=0.5)

# Initialize YouTube video ID in the session state
if "yt_video_id" not in st.session_state:
    st.session_state.yt_video_id = None

# Process the YouTube URL when the button is clicked
if process_url_clicked:
    # Use regular expression to extract video ID from the URL
    pattern = re.compile(r'(?:https?://)?(?:www\.)?(?:youtube\.com/.*(?:\?v=|/videos/|/embed/|/watch\?v=)|youtu\.be/)([^"&?/\s]{11})', re.IGNORECASE)
    match = pattern.search(url)

    try:
        if match:
            st.session_state.yt_video_id = match.group(1)
            # Load video data using YoutubeLoader
            loader = YoutubeLoader(video_id=st.session_state.yt_video_id)
            main_placeholder.text('Getting data from the video✅✅')

            docs = loader.load()
            combined_docs = [doc.page_content for doc in docs]
            text = " ".join(combined_docs)
            main_placeholder.text('Splitting text from the data✅✅')

            # Use RecursiveCharacterTextSplitter to split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
            splits = text_splitter.split_text(text)

            # Use OpenAIEmbeddings to get embeddings for the text
            embeddings = OpenAIEmbeddings()
            main_placeholder.text('Building vector database from the Data✅✅')

            # Create FAISS vector index from the text and embeddings
            vectordb = FAISS.from_texts(splits, embeddings)
            vectordb.save_local("faiss_vectordb")
            main_placeholder.text('Stored vector database locally✅✅')

    except NoTranscriptFound as e:
        main_placeholder.text('Error:\nTranscripts are not available for the provided video.')
        # Download audio and transcribe using AssemblyAI
        # ...

    except Exception as e:
        main_placeholder.text('Error:\nThere was an issue processing the video.')

# Check if YouTube video ID exists in the session state
if st.session_state.yt_video_id:
    query = main_placeholder.text_input("Question:")

    # Answer the user's question
    if st.button("Ask"):
        if query:
            embeddings = OpenAIEmbeddings()
            vectors = FAISS.load_local("faiss_vectordb", embeddings)
            st.text('Getting your answer✅✅')
            
            # Use RetrievalQA to get the answer
            chain = RetrievalQA.from_llm(llm=llm, retriever=vectors.as_retriever())
            result = chain.run(query)
            
            # Display the answer
            st.header("Answer")
            st.write(result)
        else:
            main_placeholder.text("Please enter a valid question.")

    # Add a button to clear the question
    if st.button("Clear Question"):
        query = ""

    # Display the current question
    st.write("Current Question:", query)
else:
    main_placeholder.text("Please enter a valid YouTube URL🔗 and click 👆🏻'Process URLs'.\nNote: Please give me 📽️video that is in English Language.\nCurrently, I'm handicapped🫠 with other languages")
    # main_placeholder.text('Note: Please give me video that has transcripts/captions in it.')
