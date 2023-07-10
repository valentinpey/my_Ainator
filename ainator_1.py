import os
import openai
import sys
import numpy as np
sys.path.append('../..')
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter

from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import Chroma #not good for large amount of data

from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

### LOADING
#loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
#pages = loader.load()
# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture03.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())


### SPLITING
#text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)

#headers_to_split_on = [ ("#", "Header 1"),("##", "Header 2"),("###", "Header 3"),]
#markdown_splitter = MarkdownHeaderTextSplitter( headers_to_split_on=headers_to_split_on)



#text_splitter = CharacterTextSplitter(separator="\n",chunk_size=1000,chunk_overlap=150,length_function=len)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""] #(?<=\. ) this is to specify that we want the point to be in the previous trucks not the new one
)
docs = text_splitter.split_documents(pages)

###STORAGE


embedding = OpenAIEmbeddings()
"""
sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"
embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)
np.dot(embedding1, embedding2)#dot() calcul la resembelance des deux embedings
"""
persist_directory = 'docs/chroma/'
!rm -rf ./docs/chroma  # remove old database files if any

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory=persist_directory
)
question = "is there an email i can ask for help"

#WHAT are the mos importnat part of the file

docs_1 = vectordb.similarity_search(question,k=3)#this search in our data the information by looking at the most similar info compared to the question 
#this can be problematic has the model is taking all the question sentence for the similarity research, hence we have to be very precise in the question
docs_2=vectordb.max_marginal_relevance_search(question,k=2, fetch_k=3) #first we retrieve the fetch_k and the we retrieve the k most diverse.
# we can also add a filter on the data we are using: filter={"source":"ma_source"}

docs[0].page_content#this shows the content we want

for doc in docs: #on affiche quelle sont les pdf qui semble détenir l'information
    print(doc.metadata)










