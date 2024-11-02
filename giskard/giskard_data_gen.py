from dotenv import load_dotenv
import warnings
from llama_parse import LlamaParse
import nest_asyncio;
import os
from llama_index.core.node_parser import SentenceSplitter
import pandas as pd
from giskard.rag import KnowledgeBase, generate_testset, QATestset
import giskard
from giskard.llm.client.openai import OpenAIClient
from giskard.rag import evaluate, RAGReport
from giskard.rag.metrics.ragas_metrics import (ragas_context_recall, ragas_context_precision, ragas_faithfulness,
                                               ragas_answer_relevancy)
nest_asyncio.apply()

warnings.filterwarnings('ignore')
_ = load_dotenv()

# parse pdf using llama_parse
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pdf_name = "data/pb116349-business-health-select-handbook-1024-pdfa.pdf"
# set up parser
parser = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type="markdown")

documents = parser.load_data(pdf_name)

# split and index documents
splitter = SentenceSplitter(chunk_size=1024)

nodes = splitter.get_nodes_from_documents(documents)

knowledge_base_df = pd.DataFrame([node.text for node in nodes], columns=["text"])

# WORKS
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

giskard.llm.set_llm_api("openai")

gpt4o_mini = OpenAIClient(model="gpt-4o-mini")

# retriever
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
)
ctx = StorageContext.from_defaults(persist_dir="./storage_insurance")
index = load_index_from_storage(ctx)
query_engine = index.as_query_engine()

giskard.llm.set_default_client(gpt4o_mini)

knowledge_base = KnowledgeBase(knowledge_base_df,
                               llm_client = giskard.llm.set_default_client(gpt4o_mini))

test_set = generate_testset(knowledge_base,
                           num_questions=12,
                           agent_description="A chatbot answering questions about the insurance policy document.",)
df_testset = test_set.to_pandas()

df_testset['question_type']=df_testset['metadata'].apply(lambda x: x['question_type'])

df_testset['question_type'].unique()

df_testset.groupby(['question_type'])['question'].count()

# print the df_testset values
print("generated test set is ", df_testset)


def answer_fn(question):
    answer = query_engine_gpt4o.query(question)
    return str(answer)


report = evaluate(answer_fn,
                  testset=test_set,
                  knowledge_base=knowledge_base,
                  metrics=[ragas_context_recall, ragas_context_precision, ragas_faithfulness, ragas_answer_relevancy])
