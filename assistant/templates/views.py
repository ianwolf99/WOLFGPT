from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import ChatForm
import os
import sys
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.vectorstores import Chroma
import markdown2
from django.views.decorators.csrf import csrf_exempt
# Set the directory path where your PDF files are located
PDF_DIRECTORY = os.path.join(os.path.dirname(__file__), 'pdf')  # Assuming 'pdf' directory is located in the same directory as views.py

PERSIST = True
@csrf_exempt
def chat_view(request):
    response = None
    chat_history = request.session.get('chat_history', [])

    if 'new_chat' in request.GET:
        # Handle creating a new chat session
        request.session.flush()  # Clear existing session data
        return redirect('chat')  # Redirect back to the chat page with a fresh session

    if request.method == 'POST':
        form = ChatForm(request.POST)
        if form.is_valid():
            query = form.cleaned_data['input_text']
            print("Received query:", query)  # Debug print statement for query
            
            if PERSIST and os.path.exists("persist"):
                vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
                index = VectorStoreIndexWrapper(vectorstore=vectorstore)
            else:
                if not os.path.exists(PDF_DIRECTORY):
                    os.makedirs(PDF_DIRECTORY)
                loader = DirectoryLoader(PDF_DIRECTORY)
                if PERSIST:
                    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
                else:
                    index = VectorstoreIndexCreator().from_loaders([loader])

            chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(model="gpt-3.5-turbo"),
                retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
            )

            chat_history_tuples = [(message[0], message[1], "farmer") for message in chat_history]
            result = chain({"question": query, "chat_history": chat_history_tuples})

            response = markdown2.markdown(result['answer'])

            chat_history.append((query, response, "assistant"))  # Append the sender info
            request.session['chat_history'] = chat_history
    else:
        form = ChatForm()

    print("Chat History:", chat_history)

    return render(request, 'chat.html', {'form': form, 'response': response, 'chat_history': chat_history})
