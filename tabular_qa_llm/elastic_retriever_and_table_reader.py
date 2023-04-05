from haystack import Document
import pandas as pd

import logging

import os
import time
from haystack.document_stores import ElasticsearchDocumentStore

from haystack.nodes.retriever import EmbeddingRetriever

from haystack.nodes import TableReader
from haystack.utils import print_answers

# inputs
query = "京都大学情报学的考试科目有哪些？"
doc_path = "~/Downloads/ryosuke_dev_qa_1.csv"

# logging
logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

# indexing data
df = pd.read_csv(doc_path)
# Minimal cleaning
df.fillna(value="", inplace=True)
document = Document(content=df, content_type="table")

# connecting to Elasticsearch
document_index = "document"
# local host elastic search
document_store = ElasticsearchDocumentStore(
    username="admin", password="admin", index=document_index)

# add the table to the DocumentStore
document_store.write_documents([document], index=document_index)

# initialize retriever
retriever = EmbeddingRetriever(
    document_store=document_store, embedding_model="deepset/all-mpnet-base-v2-table")
# add table embeddings to the table in DocumentStore
document_store.update_embeddings(retriever=retriever)
# retrieve relavant documents
retrived_tables = retriever.retrieve(query, top_k=5)
print(retrived_tables[0].content)

# initialize reader
reader = TableReader(
    model_name_or_path="google/tapas-base-finetuned-wtq", max_seq_len=512)
# get prediction from reader
prediction = reader.predict(query=query, documents=[document])
print_answers(prediction, details="all")
