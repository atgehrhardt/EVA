# E.V.A. (Enhanced Virtual Assistant)
Eva is an AI assistant powered by Ollama. The goal is to create a completely local LLM powered assistant with the ability to maintain long term 
memory. The long term goal of this project is to have an assistant that can listen, speak, and take actions independently.

## Current Features
- Terminal based text chat (Text Mode)
- Terminal based Voice to text chat (Voice Mode)
- Short Term Memory by surfacing recent chromadb entries as chat history
- Long Term Memory by using RAG QA Retrieval to surface data potentially outside of the chat history

#### *IMPORTANT NOTE*
To maximize speed of this application, you will need to have Ollama v0.1.33 or higher and you will need to set the below environmental variables:
```
OLLAMA_NUM_PARALLEL=3 
OLLAMA_MAX_LOADED_MODELS=3
```
This is NOT required, but will speed up the application significantly. You will need to quit and restart Ollama when setting any environmental variables

## Roadmap
##### Key:
🔲 = Not Started<br>
🔷 = In Progress<br>
✅ = Complete<br>

### Tasks
🔷 **Implement self-editing memory**<br>
&nbsp;&nbsp;&nbsp;&nbsp;🔷 Implement a max token limit for Chromadb<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🔷 Determine best size/sizes to ensure adequate memory/performance balance<br>
&nbsp;&nbsp;&nbsp;&nbsp;🔲 Implement a relevancy/time-based decay mechanism that starts deleting data after the max token limit is reached<br>
&nbsp;&nbsp;&nbsp;&nbsp;🔲 Implement continual memory summarization: Should summarize and delete data from the Chromadb in chunks (never encroaching on the chat history)<br>
🔲 **Implement text to voice**<br>
&nbsp;&nbsp;&nbsp;&nbsp;🔲 Phase 1: Voice synthesis based on completed response<br>
&nbsp;&nbsp;&nbsp;&nbsp;🔲 Phase 2: Voice synthesis in chunks based on streaming response<br>
🔲 **Implement tool usage history and the other for long term memory**<br>
🔲 **Dockerize entire application** (with GPU support for all needed libraries: Ollama, Faster-Whisper, TBD for voice synthesis)<br>
🔲 **Refactor to an API** to actually make this useful as more than a research/development tool<br>
🔲 **Implement threading/multi-tasking** for mem management and other tasks to increase speed. Just general optimization needs to be done here.<br>

### Bugs
🔲 **Recent Chat History seems to only be not grabbing the most recent chat message**<br>