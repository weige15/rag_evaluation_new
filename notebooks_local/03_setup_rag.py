# %%
"""
# RAG

In this notebook, we will setup a single Retrieval Augmented Generation model on the wiki-text dataset using DSPy, Chroma DB for vector similiarity search and OPENAI API for text generation.
"""

# %%
import dspy
import os

from langchain.text_splitter import SentenceTransformersTokenTextSplitter
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction, SentenceTransformerEmbeddingFunction

# %%
os.chdir('../')

# %%
from src.utils import *

# %%
# os.environ['OPENAI_API_KEY'] = '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'

# %%
# Load the model
# turbo = dspy.OpenAI(model='gpt-3.5-turbo')

# %%
from llama_cpp import Llama

# %%
llm_q4 = Llama(
      model_path="../llama.cpp/models/Llama-3-Instruct-8B-SPPO-Iter3-Q4_K_M.gguf",
      n_gpu_layers=-1,
      n_ctx=0,
)

llm_q4.verbose = False

# %%
import dspy

llamalm = dspy.LlamaCpp(model="llama", llama_model=llm_q4,  model_type="chat", temperature=0.4)
dspy.settings.configure(lm=llamalm)


#Define a simple signature for basic question answering
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

#Pass signature to Predict module
generate_answer = dspy.Predict(BasicQA)

# Call the predictor on a particular input.
question='What is the color of the sky?'
pred = generate_answer(question=question)

print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")


# %%
# Read the text
with open('./data_small/raw/test.txt', 'r') as f:
    text = f.read().strip()

dspy.settings.configure(lm=llamalm)

# %%
"""
----
"""

# %%
"""
## ChromaDB
"""

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=256,
    chunk_overlap=0
)
character_split_texts = character_splitter.split_text(text)

print(f"\nTotal chunks: {len(character_split_texts)}\n")

# %%
token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

print(f"\nTotal chunks: {len(token_split_texts)}")

# %%
token_split_texts[1]

# %%
token_split_texts[5]

# %%
token_split_texts

# %%
embedding_function = SentenceTransformerEmbeddingFunction()


print("Length of embedding:")
print(len(embedding_function([token_split_texts[0]])[0]))


# %%
chroma_client = chromadb.PersistentClient("chroma.db")

# %%
# Create a new collection

chroma_collection = chroma_client.get_or_create_collection("test-overlap-small-0", embedding_function=embedding_function)

ids = [str(i) for i in range(len(token_split_texts))]

# %%
chroma_collection.add(ids=ids, documents=token_split_texts)

# %%
chroma_client.list_collections()

# %%
chroma_collection.peek(1)

# %%
"""
----
"""

# %%
query = "Who was Robert Boulter?"

results = chroma_collection.query(query_texts=[query], n_results=2)
retrieved_documents = results['documents'][0]

print(f"Query: {query}")

print(f"\nRetrieved {len(retrieved_documents)} documents\n")

for docs in retrieved_documents:
    print(word_wrap(docs))


# %%
query = "Who was Du Fu?"

results = chroma_collection.query(query_texts=[query], n_results=2)
retrieved_documents = results['documents'][0]

print(f"Query: {query}")

print(f"\nRetrieved {len(retrieved_documents)} documents\n")

for docs in retrieved_documents:
    print(word_wrap(docs))

# %%
query = "When was Robert Boulter active?"

results = chroma_collection.query(query_texts=[query], n_results=2)
retrieved_documents = results['documents'][0]

print(f"Query: {query}")

print(f"\nRetrieved {len(retrieved_documents)} documents\n")

for docs in retrieved_documents:
    print(word_wrap(docs))

# %%
query = "When was Robert Boulter active?"

results = chroma_collection.query(query_texts=[query], n_results=3)
retrieved_documents = results['documents'][0]

print(f"Query: {query}")

print(f"\nRetrieved {len(retrieved_documents)} documents\n")

for docs in retrieved_documents:
    print(word_wrap(docs))

# %%
# turbo = dspy.OpenAI(model='gpt-3.5-turbo')

# dspy.settings.configure(lm=turbo)

# %%
llamalm = dspy.LlamaCpp(model="llama", llama_model=llm_q4,  model_type="chat", temperature=0.4)
dspy.settings.configure(lm=llamalm)

# %%
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Explain with words between 1 and 5 words")

# %%
# Modifying the default RAG module because it doesn't work with the SentenceTransformerEmbeddingFunction
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.chroma_collection = chroma_client.get_collection("test")
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.num_passages = num_passages
    
    def forward(self, question):
        context = self.chroma_collection.query(query_texts=[question], n_results=self.num_passages)
        context = context['documents']
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

# %%
os.environ['LANGFUSE_SECRET_KEY'] = 'sk-lf-6f3542d6-53e7-4fd2-b417-e6e2fc0512a0'
os.environ['LANGFUSE_PUBLIC_KEY'] = 'pk-lf-3d36f7c6-2840-40d1-b129-63e075e24226'
os.environ["LANGFUSE_HOST"] = 'https://us.cloud.langfuse.com'

# %%
rag = RAG(num_passages=3)

# %%
question = "Who was Robert Boulter?"
rag(question)

# %%
llamalm.inspect_history(n=1)

# %%
"""
----
"""

# %%
"""
### Using the modified ChromaDBRM
"""

# %%
from src import chromadb_rm

# %%
"""
source: https://github.com/weige15/amazon-bedrock/blob/main/dspy/dspy_bedrock.ipynb 

import chromadb
from chromadb.utils import embedding_functions
from dspy.retrieve.chromadb_rm import ChromadbRM
persist_dir="localdb"
chroma_client = chromadb.PersistentClient(path=persist_dir)
coll_name = "cuad"

#Remove any existing collection
try:
    chroma_client.delete_collection(name=coll_name)
except:
    pass

embedding_model_name = "multi-qa-MiniLM-L6-cos-v1"
#embedding_model_name = "all-mpnet-base-v2"
#embedding_model_name = "all-MiniLM-L6-v2"

sentence_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name = embedding_model_name)
cuad_db = chroma_client.get_or_create_collection(name=coll_name, embedding_function=sentence_ef,metadata={"hnsw:space":"cosine"})
"""

# %%
from chromadb.utils import embedding_functions
embedding_model_name = "multi-qa-MiniLM-L6-cos-v1"

sentence_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name = embedding_model_name)

# %%
from dspy.retrieve.chromadb_rm import ChromadbRM

chroma_rm = ChromadbRM(collection_name="test_small", persist_directory="chroma.db", embedding_function=sentence_ef,k=2)
                                   
                                   #openai_api_key=os.environ["OPENAI_API_KEY"]

# %%
dspy.settings.configure(lm=llamalm, rm=chroma_rm)

# %%
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

# %%
rag = RAG(num_passages=3)
question = "Who was Robert Boulter?"
rag(question)

# %%
from dspy.datasets import HotPotQA

# Load the dataset.
dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)

# Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

len(trainset), len(devset)

# %%
from dspy.teleprompt import BootstrapFewShot

# Validation logic: check that the predicted answer is correct.
# Also check that the retrieved context does actually contain that answer.
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM

# Set up a basic teleprompter, which will compile our RAG program.
teleprompter = BootstrapFewShot(metric=validate_context_and_answer)

# Compile!
compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

# %%
# Ask any question you like to this simple RAG program.
my_question = "Who was Robert Boulter?"

# Get the prediction. This contains `pred.context` and `pred.answer`.
pred = compiled_rag(my_question)

# Print the contexts and the answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")
print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")

# %%
llamalm.inspect_history(n=1)

# %%
for name, parameter in compiled_rag.named_predictors():
    print(name)
    print(parameter.demos[0])
    print()

# %%
"""
----
"""