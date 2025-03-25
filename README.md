# Langchain-Question-Answering-Application


## Overview
This application allows users to upload documents (PDF, DOCX, TXT) and use Langchain with Google's Gemini AI to answer questions based on the document's content. The application supports chunking, embedding, and retrieval for efficient querying.

## Features
- Upload documents in **PDF, DOCX, or TXT** formats.
- Chunk documents into smaller sections for better processing.
- Generate embeddings using **GoogleGenerativeAIEmbeddings**.
- Store embeddings using **Chroma vector database**.
- Retrieve answers using **similarity search** and **LLM inference**.
- Track and display chat history.

## Prerequisites
- Python 3.8+
- Required Python libraries (install using `requirements.txt`):
  pip install -r requirements.txt
  ```

## Installation
1. Clone the repository:
   git clone https://github.com/your-username/langchain-qa-app.git
   cd langchain-qa-app
   ```
2. Install dependencies:
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your Google API Key:
   GOOGLE_API_KEY=your_google_api_key_here
   ```
4. Run the application:
   streamlit run app.py
   ```

## Usage
1. Open the Streamlit app.
2. Enter your Google API Key in the sidebar.
3. Upload a document.
4. Click the **Add Data** button to process the file.
5. Ask questions in the provided input field.
6. View responses and chat history.

## File Structure
```
langchain-qa-app/
│── app.py                  # Main Streamlit application
│── requirements.txt         # Dependencies
│── README.md                # Project documentation
│── .env                     # API keys (excluded from Git)
│── chroma_db/               # Vector database storage
│── utils.py                 # Helper functions (optional)
```

## Technologies Used
- **Streamlit** - Web UI framework
- **Langchain** - LLM-powered question answering
- **Google Generative AI** - Embeddings and LLM
- **ChromaDB** - Vector database for document retrieval

## Contributing
Feel free to submit issues or pull requests. Contributions are welcome!

## License
This project is licensed under the MIT License.

