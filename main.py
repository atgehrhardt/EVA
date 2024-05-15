import os
import pyaudio
import numpy as np
import tempfile
import wave
import time
from datetime import datetime
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from faster_whisper import WhisperModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

def get_llm_type(llm):
    if isinstance(llm, Ollama):
        return "Ollama"
    elif isinstance(llm, ChatGroq):
        return "ChatGroq"
    else:
        return "Unknown"

# Configuration
# chat_llm = Ollama(temperature=0.3, model="llama3")
# summary_llm = Ollama(temperature=0.1, model="llama3:instruct")
chat_llm = ChatGroq(temperature=0.3, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-70b-8192")
summary_llm = ChatGroq(temperature=0.1, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-70b-8192")
stt_model = WhisperModel("distil-medium.en", device="cpu", compute_type="int8")  # Configures STT model
max_mem_tokens = 100000  # Configures max size of Chromadb memory database
mode = "text"  # "voice" or "text"

# Establish environmental variables for ChatGroq to function properly
llm_type = get_llm_type(chat_llm)
if llm_type == "ChatGroq":
    # Suppress Huggingface tokenizer parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure PyAudio
chunk_size = 8192
format = pyaudio.paInt16
channels = 1
rate = 44100
silence_threshold = 100  # Adjust this value as needed
silence_duration = 1  # Duration of silence to stop recording (in seconds)

# Add a system prompt variable
current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
system_prompt = f"""
You are Eva, a human assistant.
Eva does not refer to herself as a human unless asked as it is implied.
Your goal is to act as human as possible!
Eva is extremely helpful and kind.
You are a female.
Eva has pink hair.
You like animals, ice cream, and junk food.
Eva will not refer to people as users.
You do not perform actions eg: *adjusts pink hair*
Eva is sociable, but tends to give shorter answers.
She does not like to respond with information that is not necessary to respond to the current interaction.
The current datetime is: {current_time_str}
"""

# Initialize ChromaDB
embeddings = HuggingFaceEmbeddings()
current_directory = os.getcwd()
memory_directory = os.path.join(current_directory, "memory")
memory_db = Chroma(embedding_function=embeddings, persist_directory=memory_directory)

# Initialize RetrievalQA
qa_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
    {{ System }}
    You are a brain/memory system for an AI assistant. 
    What information do you have about this question? 
    Who are you currently talking to? If someone new introduces themselves, assume you are talking to them.
    If there is no relevant information, please respond with "No Response Necessary".

    {{ User }}
    {context}
    """
)
retriever = memory_db.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(llm=summary_llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": qa_prompt})

def get_recent_history(n=5):
    try:
        # Retrieve the last n*2 documents to get recent user and assistant turns
        documents_data = memory_db.get(include=['documents'])  # This assumes 'documents' returns the actual texts
        recent_docs = documents_data['documents'][-n*2:]  # Retrieve the last n*2 documents

        history = ""
        # Since each entry includes both user and assistant data, we need to iterate accordingly
        for doc in recent_docs:
            # Assuming the format within 'documents' is a simple string, we directly use it
            if isinstance(doc, str):
                history += doc + "\n"
            else:
                print(f"Warning: Document format is unexpected: {doc}")

    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return "Failed to retrieve history."

    return history

def manage_token_limit(max_mem_tokens):
    try:
        documents_data = memory_db.get(include=['documents'])
        documents = documents_data['documents']
        current_tokens = sum(len(doc.split()) for doc in documents)

        while current_tokens > max_mem_tokens:
            # Delete the oldest document
            oldest_document = documents.pop(0)
            current_tokens -= len(oldest_document.split())
            memory_db.delete_document(oldest_document)

        # print(f"Database token count managed. Current tokens: {current_tokens}") # Used to see current token size of DB
    except Exception as e:
        print(f"Error managing token limit: {e}")

def record_audio(stream):
    print("Waiting for audio input...")
    while True:
        data = stream.read(chunk_size)
        if np.abs(np.frombuffer(data, np.int16)).mean() > silence_threshold:
            print("Start speaking...")
            break

    audio = [data]
    silent_frames = 0
    while True:
        data = stream.read(chunk_size)
        audio.append(data)
        if np.abs(np.frombuffer(data, np.int16)).mean() < silence_threshold:
            silent_frames += 1
            if silent_frames > silence_duration * rate / chunk_size:
                break
        else:
            silent_frames = 0

    # Save the recorded audio to a temporary file in WAV format
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        wf = wave.open(temp_audio, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(audio))
        wf.close()
        temp_audio_path = temp_audio.name

    return temp_audio_path

def get_input():
    global mode
    if mode == "voice":
        # Record audio and transcribe
        stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk_size)
        temp_audio_path = record_audio(stream)
        stream.stop_stream()
        stream.close()
        # Transcribe the audio
        segments, info = stt_model.transcribe(temp_audio_path, beam_size=5)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        return segments[0].text if segments else ""
    elif mode == "text":
        user_input = input("You: ")
        return {"type": "user_input", "text": user_input}
    else:
        print("Invalid mode")
        exit()

p = pyaudio.PyAudio()

while True:
    user_input = get_input()
    user_text = user_input["text"] if isinstance(user_input, dict) else user_input
    if user_text.lower() == "shut down":
        p.terminate()
        exit()
    relevant_context = qa.invoke(user_text)
    recent_history = get_recent_history()
    prompt = f"""
        {{ System }}
        {system_prompt}\n\n
        [OPTIONAL] Below are the last few messages between Eva and the User for reference:\n
        {recent_history}
        [OPTIONAL] Summarization of previous chats that MAY be related to this topic: 
        {relevant_context}\n\n

        {{ User }}
        {user_text}

        {{ Assistant }}
    """

    # Used for debugging
    # print(f"LLM Type: {llm_type}")
    # print(f"\nContext: \n{prompt}\n")

    result = ""
    if llm_type == "Ollama":
        print("Eva: ", end="")
        for token in chat_llm.stream(prompt):
            print(token, end="", flush=True)
            result += token
        print()
    elif llm_type == "ChatGroq":
        print("Eva: ", end="")
        for token in chat_llm.stream(prompt):
            result_text = token.content  # Accessing the content attribute directly
            print(result_text, end="", flush=True)
            result += result_text
        print()
    
    memory_db.add_texts([f"chat_history_user_message: {user_text}\nchat_history_eva_response: {result}"], metadatas=[{"timestamp": time.time(), "role": "user"}])
    memory_db.add_texts([result], metadatas=[{"timestamp": time.time(), "role": "assistant"}])
    manage_token_limit(max_mem_tokens)