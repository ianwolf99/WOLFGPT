# views.py
from django.shortcuts import render
from django.http import HttpResponse
from .forms import ChatForm
import os
import sys
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.vectorstores import Chroma
import markdown2

# Set the directory path where your PDF files are located
PDF_DIRECTORY = os.path.join(os.path.dirname(__file__), 'pdf')  # Assuming 'pdf' directory is located in the same directory as views.py

#os.environ["OPENAI_API_KEY"] = constants.APIKEY

PERSIST = True

def chat_view(request):
    response = None
    if request.method == 'POST':
        form = ChatForm(request.POST)
        if form.is_valid():
            query = form.cleaned_data['input_text']
            chat_history = request.session.get('chat_history', [])
            if PERSIST and os.path.exists("persist"):
                vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
                index = VectorStoreIndexWrapper(vectorstore=vectorstore)
            else:
                if not os.path.exists(PDF_DIRECTORY):
                    # Create the directory if it doesn't exist
                    os.makedirs(PDF_DIRECTORY)
                loader = DirectoryLoader(PDF_DIRECTORY)
                if PERSIST:
                    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders(
                        [loader])
                else:
                    index = VectorstoreIndexCreator().from_loaders([loader])

            chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(model="gpt-3.5-turbo"),
                retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
            )

            # Convert chat history to list of tuples
            chat_history_tuples = []
            for message in chat_history:
                chat_history_tuples.append((message[0], message[1]))

            result = chain({"question": query, "chat_history": chat_history_tuples})
            response = markdown2.markdown(result['answer'])


            # Append the query and response to chat history
            chat_history.append([query, response])
            request.session['chat_history'] = chat_history

    else:
        form = ChatForm()

    return render(request, 'chat.html', {'form': form, 'response': response})