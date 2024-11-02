from dotenv import load_dotenv
import warnings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_parse import LlamaParse
import nest_asyncio;
import os
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI

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

embed_model = OpenAIEmbedding(model="text-embedding-3-large")

vector_index = VectorStoreIndex(nodes, embed_model = embed_model)

vector_index.storage_context.persist(persist_dir="./storage_insurance")

llm_gpt4o = OpenAI(model="gpt-4o-mini", api_key = OPENAI_API_KEY)

query_engine_gpt4o = vector_index.as_query_engine(similarity_top_k=3, llm=llm_gpt4o)

query1 = "Whats the cashback amount for optical expenses ?"
resp = query_engine_gpt4o.query(query1)
print("GPT-4o-mini:")
print(str(resp))