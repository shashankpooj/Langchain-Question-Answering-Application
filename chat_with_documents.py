import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
import os
# import os
import os
import shutil
import stat
from langchain.vectorstores import Chroma

def close_chroma():
    """Ensures Chroma database is closed before deletion."""
    try:
        vector_store = Chroma(persist_directory=os.path.abspath("./chroma_db"))
        del vector_store  # Delete reference
        import gc
        gc.collect()  # Force garbage collection
        print("ChromaDB closed successfully.")
    except Exception as e:
        print(f"Error closing ChromaDB: {e}")

def remove_directory(path):
    """Removes the directory after ensuring all files are unlocked."""
    if os.path.exists(path):
        # Change permissions to writable
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                os.chmod(file_path, stat.S_IWRITE)  # Set file to writable

        try:
            shutil.rmtree(path)
            print("Directory deleted successfully.")
        except Exception as e:
            print(f"Error deleting directory: {e}")
    else:
        print("Directory does not exist.")

# Close Chroma before trying to delete the database
close_chroma()
remove_directory("./chroma_db")  # Deletes old database




def load_document(file):
    import os
    name,extension=os.path.splitext(file)

    if extension=='.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader=PyPDFLoader(file)
    elif extension=='.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader=Docx2txtLoader(file)
    elif extension=='.txt':
        from langchain.document_loaders import TextLoader
        loader=TextLoader(file)
    else:
        print('document format is not supported')
        return None
        
          
    data=loader.load()
    return data


def chunk_data(data,chunk_size=256,chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=0)
    chunks=text_splitter.split_documents(data)
    return chunks
    


def create_embeddings(chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=Chroma.from_documents(chunks,embeddings,persist_directory=os.path.abspath("./chroma_db") )
    vector_store.persist()
    return vector_store


def ask_and_get_answer(vector_store,q,k=3):
    from langchain.chains import RetrievalQA
    from langchain_google_genai import GoogleGenerativeAI

    llm=GoogleGenerativeAI(model="gemini-2.0-flash",temperature=1)

    retriever=vector_store.as_retriever(search_type='similarity',search_kwargs={'k':k})

    chain=RetrievalQA.from_chain_type(llm=llm,chain_type='stuff',retriever=retriever)

    answer=chain.run(q)
    return answer


def calculate_embedding_costs(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: ${total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens,total_tokens/1000*0.0004


def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']



if __name__=="__main__":
    import os
    from dotenv import load_dotenv

# Manually specify the path to .env
    env_path = r"C:\Users\DEEKSHA B POOJARY\OneDrive\Documents\Udemy course\.env"
    load_dotenv(env_path, override=True)

    st.image('shashank.jpg',width=230)
    st.subheader('🔗Langchain Question Answering Application🤖')
    with st.sidebar:
        api_key=st.text_input('Google API Key:',type='password')
        if api_key:
            os.environ['GOOGLE_API_KEY']=api_key
        
        uploaded_file=st.file_uploader('Upload a file:',type=['pdf','docx','txt'])
        chunk_size=st.number_input('Chunk size:',min_value=100,max_value=2048,value=512,on_change=clear_history)
        k=st.number_input('k',min_value=1,max_value=20,value=3,on_change=clear_history)
        add_data=st.button('Add Data')


        if  uploaded_file and add_data:
            with st.spinner('Reading,chunking and embedding file...'):
                bytes_data=uploaded_file.read()
                file_name=os.path.join('./',uploaded_file.name)
                with open(file_name,'wb') as f:
                    f.write(bytes_data)

            data=load_document(file_name)
            chunks=chunk_data(data,chunk_size=chunk_size)
            st.write(f'Chunk size: {chunk_size},Chunks:{len(chunks)}')

            tokens,embedding_cost=calculate_embedding_costs(chunks)
            st.write(f'Embedding cost: ${embedding_cost:.4f}')

            vector_store=create_embeddings(chunks)

            st.session_state.vs=vector_store
            st.success('File uploaded,chunked and embedded successfully')


    # q=st.text_input('Ask a question about the content of your file:')
    # if q:
    #     if 'vs' in st.session_state:
    #         vector_store=st.session_state.vs
    #         st.write(f'k:{k}')
    #         answer=ask_and_get_answer(vector_store,q,k)
    #         st.text_area('LLM Answer:',value=answer)

    

    # st.divider()
    # if 'history' not in st.session_state:
    #     st.session_state.history=''
    # value=f'Q: {q} \nA: {answer}'
    # st.session_state.history=f'{value} \n {"-" * 100} \n {st.session_state.history}'
    # h=st.session_state.history
    # st.text_area(label='Chat History',value=h,key='history',height=400)
            
    q = st.text_input('Ask a question about the content of your file:')
    answer = ""  # Define answer before using it

    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, q, k)
            st.text_area('LLM Answer:', value=answer)

            st.divider()
            if 'history' not in st.session_state:
                st.session_state.history = ''

        # Ensure answer is not empty before adding to history
            if answer:
                value = f'Q: {q} \nA: {answer}'
                st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'

            h = st.session_state.history
            st.text_area(label='Chat History', value=h, key='history', height=400)



