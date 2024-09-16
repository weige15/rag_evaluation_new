# %%
"""
Evaluation of Synthetic Dataset
===

Now, that we have generated a synthetic dataset and also built a RAG pipeline, let's first evaluate how good our dataset is. Then, we will filter out a gold dataset and then evaluate the RAG pipeline on the gold dataset.
"""

# %%
import os
import dspy
import json

# %%
os.chdir('../')

# %%
DATASET_FPATH = './data_small/processed/dataset.json'

# %%
# Read the dataset.
with open(DATASET_FPATH, 'r') as f:
    dataset = json.load(f)

# %%
dataset.keys()

# %%
# Print an example from each key of dataset
for key in dataset.keys():
    print(f"{key}:")
    for k,v in dataset[key].items():
        print(f"\t{k}: {v}")
        break
    print()



# %%
"""
RAGAS
---
"""

# %%
from ragas import evaluate

# %%
import pandas as pd
# Creating the DataFrame
data = []
for query_id, query_text in dataset['queries'].items():
    answer_text = dataset['answers'].get(query_id)
    doc_ids = dataset['relevant_docs'].get(query_id, [])
    for doc_id in doc_ids:
        corpus_text = dataset['corpus'].get(doc_id)
        # Rename ['question', 'ground_truth', 'answer', 'contexts']

        # data.append({"query": query_text, "answer": answer_text, "corpus": corpus_text})
        data.append({"question": query_text, "ground_truths": [answer_text], "answer": answer_text, "contexts": [corpus_text]})

df = pd.DataFrame(data)
df.head()

# %%
df.to_csv('./data_small/processed/synthetic_dataset.csv', index=False)

# %%
from datasets import Dataset
ds = Dataset.from_pandas(df)

# %%
#os.environ['OPENAI_API_KEY'] = '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'

# %%
os.environ['LANGFUSE_SECRET_KEY'] = 'sk-lf-6f3542d6-53e7-4fd2-b417-e6e2fc0512a0'
os.environ['LANGFUSE_PUBLIC_KEY'] = 'pk-lf-3d36f7c6-2840-40d1-b129-63e075e24226'
os.environ["LANGFUSE_HOST"] = 'https://us.cloud.langfuse.com'

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
question='What is the color of the ocean?'
pred = generate_answer(question=question)

print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")


# %%
llamalm

# %%
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

# %%
template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate.from_template(template)



# %%
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# %%
# Make sure the model path is correct for your system!
langchain_llm = LlamaCpp(
    model_path="../llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

# %%
question = """
Question: A rap battle between Stephen Colbert and John Oliver
"""
langchain_llm.invoke(question)

# %%
from langchain_community.embeddings import LlamaCppEmbeddings

# %%
langchain_embeddings = LlamaCppEmbeddings(model_path="../llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
                              n_ctx=2048,
                              n_gpu_layers=-1
                            )

# %%
# langchain_embeddings = LlamaCppEmbeddings(model_path="../llama.cpp/models/Llama-3-Instruct-8B-SPPO-Iter3-Q4_K_M.gguf")

# %%
text = "This is a test document."

query_result = langchain_embeddings.embed_query(text)

doc_result = langchain_embeddings.embed_documents([text])

# %%
doc_result

# %%
query_result

# %%
langchain_embeddings

# %%
from ragas.metrics import faithfulness
from ragas import evaluate

results = evaluate( dataset = ds,metrics=[faithfulness], llm=langchain_llm, embeddings=langchain_embeddings)

# %%
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

# langchain_llm =  # any langchain LLM instance
# langchain_embeddings = # any langchain Embeddings instance

results = evaluate(metrics=[], llm=langchain_llm, embeddings=langchain_embeddings)

# %%
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

# define llm and embeddings
langchain_llm = BaseLanguageModel(model=llamalm) # any langchain LLM instance
langchain_embeddings = Embeddings(model=llamalm) # any langchain Embeddings instance

# make sure to wrap them with wrappers
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

langchain_llm = LangchainLLMWrapper(langchain_llm)
langchain_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

# you can also use custom LLMs and Embeddings here but make sure 
# they are subclasses of BaseRagasLLM and BaseRagasEmbeddings
llm = MyCustomLLM()
embeddings = MyCustomEmbeddings()

# %%


# %%
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
    context_precision,
)
try:
    result = evaluate(
        dataset = ds,
        llm = llamalm,
        metrics=[
            context_relevancy,
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        raise_exceptions=False
    )
except Exception as e:
    print(e)

# %%
# from ragas.metrics import (
#     answer_relevancy,
#     faithfulness,
#     context_recall,
#     context_precision,
#     answer_similarity,
#     context_relevancy
# )

# result = evaluate(
#     ds,
#     metrics=[
#         context_precision,
#         faithfulness,
#         answer_relevancy,
#         context_recall,
#         answer_similarity,
#         context_relevancy
#     ],
# )

# %%
result

# %%
result.to_pandas().head()

# %%
# Use the save_result function to save the result to a csv file.
import time

def save_result(result):
    exp_name = f"results/eval_synthetic_data_{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"Saving results to {exp_name}.csv")
    # make dir results
    if not os.path.exists('results'):
        os.makedirs('results')

    # Write to file
    result.to_pandas().to_csv(f"{exp_name}.csv")

# %%
# Uncomment the following line to save the result.
save_result(result)

# %%
os.environ['WANDB_NOTEBOOK_NAME'] = '04_eval_synth_data.ipynb'

# %%
os.environ['WANDB_API_KEY'] = '489eb28b2888d684cef50ac9633d922c62b6c655'

# %%
# Logging to wandb

import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="wikitext-rag-synthetic-eval",
    
    # track hyperparameters and run metadata
    config={
        "chuck_size": 1024,
        "sentence_chunck_overlap": 200,
        "number_of_questions": len(ds),
        "comments": "Synthetic dataset where ground truth and the answer are the same.",
    }
)

wandb.log(result)

wandb.finish()

# %%
"""
-----
"""