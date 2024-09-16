# %%
"""
# RAG Evaluations
"""

# %%
import os
import dspy

# %%
os.chdir('../')

# %%
from src.chromadb_rm import ChromadbRM

# %%
os.environ['OPENAI_API_KEY'] = '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'

# %%
os.environ['LANGFUSE_SECRET_KEY'] = 'sk-lf-6f3542d6-53e7-4fd2-b417-e6e2fc0512a0'
os.environ['LANGFUSE_PUBLIC_KEY'] = 'pk-lf-3d36f7c6-2840-40d1-b129-63e075e24226'
os.environ["LANGFUSE_HOST"] = 'https://us.cloud.langfuse.com'

# %%
class GenerateAnswer(dspy.Signature):
    """Answer questions given the context"""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Short factual answer to the question. 1 - 5 words long.")

class RAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

# %%
def setup():
    """
    Setup the dsypy and retrieval models
    """

    turbo = dspy.OpenAI(model='gpt-3.5-turbo')

    chroma_rm = ChromadbRM(collection_name="test-overlap-0", persist_directory="chroma.db", local_embed_model="sentence-transformers/paraphrase-MiniLM-L6-v2",
                                   openai_api_key=os.environ["OPENAI_API_KEY"])

    dspy.settings.configure(lm=turbo, rm=chroma_rm)
    
    rag = RAG()

    return rag

# %%
rag = setup()

# %%
# Read question, ground_truths from ./data/processed/synthetic_dataset.csv
import pandas as pd

df = pd.read_csv("./data/processed/synthetic_dataset.csv")

df = df[['question', 'ground_truths']]

# %%
df.head()

# %%
from sklearn.model_selection import train_test_split

# %%
# split the data into train and test
train, test = train_test_split(df, test_size=0.2)

# %%
# save the train and test data
train.to_csv("./data/processed/train_synthetic.csv", index=False)
test.to_csv("./data/processed/test_synthetic.csv", index=False)

# load the train and test data
train = pd.read_csv("./data/processed/train_synthetic.csv")
test = pd.read_csv("./data/processed/test_synthetic.csv")

# %%
import tqdm

# Create an empty list to store rows
eval_results_rows = []

for index, row in test.iterrows():
    # Get the question
    question = row['question']
    # Response from rag
    response = rag(question)
    # Create a dictionary to represent a row
    row_dict = {'question': question, 'contexts': response.context, 'answer': response.answer, 'ground_truths' : row['ground_truths']}
    # Append the row dictionary to the list
    eval_results_rows.append(row_dict)

# Create the df_eval_results DataFrame from the list of rows
df_eval_results = pd.DataFrame(eval_results_rows)


# %%
df_eval_results

# %%
import ast

# df_eval_results ground_truths to list
df_eval_results['ground_truths'] = df_eval_results['ground_truths'].apply(lambda x: ast.literal_eval(x))

# %%
# Save the df_eval_results DataFrame to a csv file
import time
EXP_NAME = "SIMPLE_RAG_NO_OVERLAP"
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
df_eval_results.to_csv('./results/inference_' + EXP_NAME + '_' + TIMESTAMP + '.csv', index=False)

# %%
"""
Now, that we have answers for all the questions, we can evaluate the RAG model.
"""

# %%
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
    context_precision,
)

ds = Dataset.from_pandas(df_eval_results)


try:
    result = evaluate(
        dataset = ds,
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
# from datasets import Dataset
# from ragas import evaluate

# ds = Dataset.from_pandas(df_eval_results)

# result = evaluate(
#     ds,
#     metrics=[
#         faithfulness,
#         answer_relevancy,
#         context_relevancy,
#         context_recall,
#         context_precision
#     ],
# )

# %%
result

# %%
# save the result
result.to_pandas().to_csv('./results/evaluation_' + EXP_NAME + '_' + TIMESTAMP + '.csv', index=False)

# %%
result.to_pandas()

# %%
os.environ['WANDB_NOTEBOOK_NAME'] = '05_eval_rag.ipynb'

# %%
os.environ['WANDB_API_KEY'] = '489eb28b2888d684cef50ac9633d922c62b6c655'

# %%
# Logging to wandb

import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="wikitext-rag-eval",
    
    # track hyperparameters and run metadata
    config={
        "number_of_questions": len(ds),
        "comments": "Simple QA RAG model with no teleprompter - chunk overlap size 0",
        "model": "RAG",
        "dataset": "Synthetic",
        "num_passages": 5,
        "openai_model": "gpt-3.5-turbo",
        "chroma_collection_name": "test-overlap-64",
        "chroma_persist_directory": "chroma.db",
        "chroma_local_embed_model": "sentence-transformers/paraphrase-MiniLM-L6-v2",

    }
)

wandb.log(result)

wandb.finish()

# %%
"""
----
"""

# %%
"""
Now, let's compile the RAG using teleprompters.
"""

# %%
train.reset_index(inplace=True, drop=True)

# %%
train = train[:10]

# %%
train

# %%
import ast

trainset = []
for i in range(5):
    ex = dspy.Example(
        question=train['question'].iloc[i],
        answer=ast.literal_eval(train['ground_truths'].iloc[i])[0]
    )
    ex = ex.with_inputs('question')
    trainset.append(ex)

# %%
trainset

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
import ast
def get_evals(dataset, rag):
    # Create an empty list to store rows
    eval_results_rows = []

    for index, row in dataset.iterrows():
        # Get the question
        question = row['question']
        # Response from rag
        response = rag(question)
        # Create a dictionary to represent a row
        row_dict = {'question': question, 'contexts': response.context, 'answer': response.answer, 'ground_truths' : row['ground_truths']}
        # Append the row dictionary to the list
        eval_results_rows.append(row_dict)

    # Create the df_eval_results DataFrame from the list of rows
    df_eval_results = pd.DataFrame(eval_results_rows)

    # Convert 'ground_truths' column to list
    df_eval_results['ground_truths'] = df_eval_results['ground_truths'].apply(lambda x: ast.literal_eval(x))

    return df_eval_results


# %%

df_eval_results = get_evals(test, compiled_rag)


# %%
# Save the df_eval_results DataFrame to a csv file
import time
EXP_NAME = "COMPILED_RAG_OVERLAP_0"
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
df_eval_results.to_csv('./results/inference_' + EXP_NAME + '_' + TIMESTAMP + '.csv', index=False)

# %%
"""
Now, that we have answers for all the questions, we can evaluate the RAG model.
"""

# %%
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
    context_precision,
)

ds = Dataset.from_pandas(df_eval_results)


try:
    result = evaluate(
        dataset = ds,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_relevancy,
            context_recall,
            context_precision,
        ],
        raise_exceptions=False
    )
except Exception as e:
    print(e)

# %%
# ds = Dataset.from_pandas(df_eval_results)

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
# save the result
result.to_pandas().to_csv('./results/evaluation_' + EXP_NAME + '_' + TIMESTAMP + '.csv', index=False)

# %%
result.to_pandas()

# %%
# Logging to wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="wikitext-rag-eval",
     
    # track hyperparameters and run metadata(you can see that this is the "compiled version")
                                             ################################################
    config={
        "number_of_questions": len(ds),
        "comments": "Compiled QA RAG model with teleprompter - OVERLAP 0",
        "model": "RAG",
        "dataset": "Synthetic",
        "num_passages": 5,
        "openai_model": "gpt-3.5-turbo",
        "chroma_collection_name": "test",
        "chroma_persist_directory": "chroma.db",
        "chroma_local_embed_model": "sentence-transformers/paraphrase-MiniLM-L6-v2",

    }
)

wandb.log(result)

wandb.finish()

# %%
"""
-------
"""

# %%
"""
No Retrieval
---
"""

# %%
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# %%
# Define the predictor.
generate_answer = dspy.Predict(BasicQA)

# %%
eval_results_rows = []

for index, row in test.iterrows():
    # Get the question
    question = row['question']
    # Response from rag
    response = generate_answer(question = question)
    # Create a dictionary to represent a row
    row_dict = {'question': question, 'answer': response.answer, 'ground_truths' : row['ground_truths']}
    # Append the row dictionary to the list
    eval_results_rows.append(row_dict)

# Create the df_eval_results DataFrame from the list of rows
df_eval_results = pd.DataFrame(eval_results_rows)

# Convert 'ground_truths' column to list
df_eval_results['ground_truths'] = df_eval_results['ground_truths'].apply(lambda x: ast.literal_eval(x))

# %%
from datasets import Dataset
from ragas.metrics import (
    answer_similarity
)

ds = Dataset.from_pandas(df_eval_results)


try:
    result = evaluate(
        dataset = ds,
        metrics=[
            answer_similarity
        ],
        raise_exceptions=False
    )
except Exception as e:
    print(e)

# %%
# ds = Dataset.from_pandas(df_eval_results)

# result = evaluate(
#     ds,
#     metrics=[
#         answer_similarity
#     ],
# )

# %%
result

# %%
EXP_NAME = "BASIC_QA_OVERLAP_64"
# save the result
result.to_pandas().to_csv('./results/evaluation_' + EXP_NAME + '_' + TIMESTAMP + '.csv', index=False)

# %%
result.to_pandas()

# %%
# Logging to wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="wikitext-rag-eval",
    
    # track hyperparameters and run metadata
    config={
        "number_of_questions": len(ds),
        "comments": "No RAG model - just basic QA model - OVERLAP 64",
        "model": "RAG",
        "dataset": "Synthetic",
        "num_passages": 5,
        "openai_model": "gpt-3.5-turbo",
        "chroma_collection_name": "test",
        "chroma_persist_directory": "chroma.db",
        "chroma_local_embed_model": "sentence-transformers/paraphrase-MiniLM-L6-v2",

    }
)

wandb.log(result)

wandb.finish()

# %%
