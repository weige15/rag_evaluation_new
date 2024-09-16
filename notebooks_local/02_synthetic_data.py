# %%
"""
Generate Synthetic Data (Q&A)
===


In this notebook, we generate a sythetic dataset that contains all the questions and answers from the wiki-text-2 raw test dataset. By curating the dataset in this way, we can use it to evaluate the performance of our full pipeline.
"""

# %%
import os

# %%
pwd

# %%
os.chdir('../')

# %%
import json

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import MetadataMode

# %%
def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f'Loaded {len(docs)} docs')
    
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f'Parsed {len(nodes)} nodes')

    corpus = {node.node_id: node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes}
    return corpus

# %%
TRAIN_FILES = ['./data_small/raw/test.txt']
train_corpus = load_corpus(TRAIN_FILES, verbose=True)

# %%
len(train_corpus)

# %%
TRAIN_CORPUS_FPATH = './data_small/processed/corpus.json'

if not os.path.exists('./data_small'):
    os.mkdir('./data_small')

with open(TRAIN_CORPUS_FPATH, 'w+') as f:
    json.dump(train_corpus, f)

# %%
import re
import uuid

#from llama_index.llms.openai import OpenAI
from llama_index.core.schema import MetadataMode
from tqdm.notebook import tqdm

# %%
TRAIN_QUERIES_FPATH = './data_small/processed/queries.json'
TRAIN_RELEVANT_DOCS_FPATH = './data_small/processed/relevant_docs.json'
TRAIN_ANSWERS_FPATH = './data_small/processed/answers.json'

# %%
with open(TRAIN_CORPUS_FPATH, 'r+') as f:
    train_corpus = json.load(f)

# %%
# Creating a corpus of text
train_corpus = {k: train_corpus[k] for k in list(train_corpus.keys())}

# %%
# Sample 10 queries
for key, value in list(train_corpus.items())[:10]:
    print(key, value)

# %%
#os.environ['OPENAI_API_KEY'] = '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'

# %%
from llama_cpp import Llama

# %%
pwd

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
llamalm

# %%
###### Generate queries and answers #####
def generate_queries_and_answers(
    corpus,
    num_questions_per_chunk=2,
    prompt_template=None,
    verbose=False,
):
    """
    Automatically generate hypothetical questions and answers that could be 
    answered with the doc in the corpus.
    """
    # llm = OpenAI(model='gpt-3.5-turbo')
    llm = llamalm
    prompt_template = prompt_template or """\
    Context information is provided below.

    ---------------------
    {context_str}
    ---------------------

    With the provided context information and no prior knowledge,
    create {num_questions_per_chunk} question(s) and their corresponding answer(s) 
    for an upcoming quiz/examination. Answers should be concise, limited to 1-5 words. 
    The questions and answers should be diverse in nature across the document 
    and directly related to the context information."
"""


    queries = {}
    answers = {}
    relevant_docs = {}
    for node_id, text in tqdm(corpus.items()):
        query = prompt_template.format(context_str=text, num_questions_per_chunk=num_questions_per_chunk)
        #response = llm.complete(query)
        response = dspy.Predict(BasicQA)
 
        result = str(response).strip().split("\n")
        q_a_pairs = zip(result[0::2], result[1::2])  # Assuming alternating questions and answers

        for question, answer in q_a_pairs:
            question = re.sub(r"^\d+[\).\s]", "", question).strip()
            if len(question) > 0 and len(answer) > 0:
                question_id = str(uuid.uuid4())
                question = question.replace("Question:", "").strip()
                queries[question_id] = question
                answer = answer.replace("Answer:", "").strip()
                answers[question_id] = answer
                relevant_docs[question_id] = [node_id]
    
    return queries, answers, relevant_docs


# %%
train_queries, train_answers, train_relevant_docs = generate_queries_and_answers(
    train_corpus,
    num_questions_per_chunk=1,
    verbose=True,
)

# %%
with open(TRAIN_QUERIES_FPATH, 'w+') as f:
    json.dump(train_queries, f)

with open(TRAIN_ANSWERS_FPATH, 'w+') as f:
    json.dump(train_answers, f)

with open(TRAIN_RELEVANT_DOCS_FPATH, 'w+') as f:
    json.dump(train_relevant_docs, f)

# %%
TRAIN_DATASET_FPATH = './data_small/processed/dataset.json'

# %%
train_dataset = {
    'queries': train_queries,
    'answers': train_answers,
    'corpus': train_corpus,
    'relevant_docs': train_relevant_docs,
}

# %%
if os.path.exists(TRAIN_DATASET_FPATH):
    os.remove(TRAIN_DATASET_FPATH)
with open(TRAIN_DATASET_FPATH, 'w+') as f:
    json.dump(train_dataset, f)

# %%
"""
---
"""