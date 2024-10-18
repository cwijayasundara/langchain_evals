from dotenv import load_dotenv
import warnings
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
from typing import TypedDict
from pydantic import BaseModel
import asyncio
import csv
from typing import TypedDict
from time import time
import nest_asyncio

nest_asyncio.apply()

warnings.filterwarnings('ignore')
_ = load_dotenv()

embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
rag_llm = ChatOpenAI(model="gpt-4o-mini") # Used for RAG
qa_llm = ChatOpenAI(model="gpt-4o", temperature=0.1) # Used to create eval dataset
benchmark_llm = ChatAnthropic(model="claude-3-5-sonnet-20240620") # Judge LLM

loader = DirectoryLoader("data/paul_graham/", use_multithreading=True, loader_cls=TextLoader)
text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n", 
        "\n", 
        " ",
        "",
    ],
    chunk_size=3000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
documents = loader.load_and_split(text_splitter=text_splitter) # Load text
vectorstore = Chroma.from_documents(documents, embedding=embed_model, collection_name="groq_rag")
retriever = vectorstore.as_retriever()
print(f"Documents indexed: {len(documents)}")

async def query_retriever(question: str):
    return await retriever.ainvoke(question)

# Use this function to run the query
result = asyncio.run(query_retriever("What did paul graham do growing up?"))
print(result)

RAG_SYSTEM_PROMPT = """\
You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context given within delimiters to answer the human's questions.
```
{context}
```
If you don't know the answer, just say that you don't know.\
"""

RAG_HUMAN_PROMPT = "{input}"

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", RAG_HUMAN_PROMPT)
])

def format_docs(docs: List[Document]):
    """Format the retrieved documents"""
    return "\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    } 
    | RAG_PROMPT
    | rag_llm
    | StrOutputParser()

)

async def run_rag_query(question: str):
    return await rag_chain.ainvoke(question)

result = asyncio.run(run_rag_query("What did paul graham do growing up?"))
print(result)

class QAResponse(TypedDict):
    question_1: str
    question_2: str
    question_3: str

QA_HUMAN_PROMPT = """\
You are a Teacher/ Professor. Your task is to setup questions for an upcoming \
quiz/examination. The questions should be diverse in nature across the document. \
Given the context information and not prior knowledge, generate only questions based on the below context. \
Restrict the questions to the context information provided within the delimiters.
```
{text}
```
Output the questions in JSON format with the keys question_1, question_2 and question_3 \
and make sure to escape any special characters to output clean, valid JSON.\
"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("human", QA_HUMAN_PROMPT)
])

qa_chain = (
{"text": RunnablePassthrough()}
| QA_PROMPT
| qa_llm.with_structured_output(method='json_mode', schema=QAResponse)
)

texts = [doc.page_content for doc in documents]

async def generate_questions():
    texts = [doc.page_content for doc in documents]
    return await qa_chain.abatch(texts)

questions: List[Dict] = asyncio.run(generate_questions())

print(f"From document: \n{texts[0]}\n")
print(f"Questions generated:")
for i, q in enumerate(questions[0].values(), 1): print(f'{i}: {q}')

# Evaluate the RAG Pipeline
# Response object structure
class EvalResponse(BaseModel):
    score: int
    explanation: str

EVAL_HUMAN_PROMPT = """\
You are given a question, an answer and reference text within marked delimiters. \
You must determine whether the given answer correctly answers the question based on the reference text. Here is the data:
```Question
{question}
```
```Reference
{context}
```
```Answer
{answer}
```
Respond with a valid JSON object containing two fields:
{{
    "score": "int: a score between 0-10, 10 being highest, on whether the question is correctly and fully answered by the answer",
    "explanation": "str: Provide an explanation as to why the score was given."
}} 
Make sure to escape any special characters to output clean, valid JSON.\
"""

EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("human", EVAL_HUMAN_PROMPT)
])

eval_chain = (
    {
    "context": RunnablePassthrough(),
    "question": RunnablePassthrough(), 
    "answer": RunnablePassthrough(),
    }
    | EVAL_PROMPT
    | benchmark_llm.with_structured_output(schema=EvalResponse)
)

async def evaluate_rag():
    q1 = questions[-1]['question_1']
    t1 = texts[-1]

    print(f"Question: {q1}")
    a1 = await rag_chain.ainvoke(q1)
    print(f"Answer: {a1}")
    eval_input = {
        'context': t1,
        'question': q1,
        'answer': a1
    }
    response = await eval_chain.ainvoke(eval_input)
    print("---------------------")
    print(f"Score: {response.score}")
    print(f"Explanation: {response.explanation}")
    print("---------------------")

# Run the async function
asyncio.run(evaluate_rag())

class EvalResult(TypedDict): # For type hinting
    question: str
    answer: str
    context: str
    score: int # Score between 0 - 10
    explanation: str # Explanation on why the score was given

async def evaluate(questions: List[Dict] = questions, texts: List[str] = texts) -> List[EvalResult]:
    # Prepare inputs
    batch_rag_inputs: List[Dict] = []
    evals: List[Dict] = []
    for q_dict, context in zip(questions, texts): 
        for question in q_dict.values(): 
            batch_rag_inputs.append(question)
            evals.append({'question': question, 'context': context})

    print(f"Running RAG pipeline for {len(batch_rag_inputs)} questions")
    start = time()
    answers = await rag_chain.abatch(batch_rag_inputs, config={'max_concurrency': 2}) # Reduce concurrency to avoid hitting rate limits
    end = time()
    print(f"Time taken: {end - start}")

    # Update eval_input with the answers from the rag_chain
    for eval_input, answer in zip(evals, answers):
        eval_input.update({'answer': answer})
    
    # Run eval_chain to get evaluation
    print(f"Evaluating RAG pipeline...")
    start = time()
    batch_score_explanations = await eval_chain.abatch(evals, config={'max_concurrency': 2}) # Pass in eval which contains List of 'answer', 'context', 'question'
    end = time()
    print(f"Time taken: {end - start}")
    
    # Update eval variable with the score and explanation
    for eval, score_exp_dict in zip(evals, batch_score_explanations):
        eval.update({
            'score': score_exp_dict.score,
            'explanation': score_exp_dict.explanation
        })
    
    return evals

async def main():
    return await evaluate(questions[:1], texts[:1])

evaluations = asyncio.run(main())

csv_file = 'evaluations_rag.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Context', 'Score', 'Question', 'Answer', 'Explanation'])
    for eval in evaluations:
        writer.writerow([eval['context'], eval['score'], eval['question'], eval['answer'], eval['explanation']])

print(f"Evaluations saved to {csv_file}")