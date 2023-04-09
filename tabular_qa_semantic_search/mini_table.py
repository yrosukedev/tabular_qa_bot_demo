import pandas as pd

from typing import List

import logging

from haystack import Document

from haystack.document_stores import ElasticsearchDocumentStore

from haystack.nodes.retriever import EmbeddingRetriever


def flat_table_cell_to_table(table: pd.DataFrame) -> List[pd.DataFrame]:
    # Iterate over the rows of the input DataFrame
    output = list()
    for column_label, content in table.items():
        for index_label, value in content.items():
            output.append(
                pd.DataFrame([index_label, value], columns=[column_label]))
    return output


def index_data_from_table(doc_path: str, document_index: str) -> None:

    # logging
    logging.basicConfig(
        format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
    logging.getLogger("haystack").setLevel(logging.INFO)

    # indexing data
    logging.getLogger("haystack").info("🚧 start to read table as documents")
    df = pd.read_csv(doc_path, index_col=0)
    df.fillna(value="", inplace=True)  # Minimal cleaning
    tables = flat_table_cell_to_table(table=df)
    documents = [Document(content=table, content_type="table")
                 for table in tables]
    logging.getLogger("haystack").info("✅ succeeds to read table as documents")

    # connecting to Elasticsearch
    logging.getLogger("haystack").info("🚧 start to connect to Elasticsearch")
    document_store = ElasticsearchDocumentStore(
        username="admin", password="admin", index=document_index)
    logging.getLogger("haystack").info(
        "✅ succeeds to connect to Elasticsearch")

    # add the table to the DocumentStore
    logging.getLogger("haystack").info(
        "🚧 start to write documents to Elasticsearch")
    document_store.write_documents(documents, index=document_index)
    logging.getLogger("haystack").info(
        "✅ succeeds to write documents to Elasticsearch")

    # update embeddings
    retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model="deepset/all-mpnet-base-v2-table")
    logging.getLogger("haystack").info("🚧 start to update embeddings")
    document_store.update_embeddings(retriever=retriever)
    logging.getLogger("haystack").info("✅ succeeds to update embeddings")


def query_table(query: str, document_index: str) -> None:

    # logging
    logging.basicConfig(
        format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
    logging.getLogger("haystack").setLevel(logging.INFO)

    # connecting to Elasticsearch
    logging.getLogger("haystack").info("🚧 start to connect to Elasticsearch")
    document_store = ElasticsearchDocumentStore(
        username="admin", password="admin", index=document_index)
    logging.getLogger("haystack").info(
        "✅ succeeds to connect to Elasticsearch")

    # initialize retriever
    logging.getLogger("haystack").info(
        "🚧 start to retrieve relavant tables with Retriever")
    retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model="deepset/all-mpnet-base-v2-table")
    # retrieve relavant documents
    retrived_tables = retriever.retrieve(query, index=document_index, top_k=5)
    logging.getLogger("haystack").info(
        "✅ succeeds to retrieve relavant tables with Retriever")
    for table in retrived_tables:
        print(table)


if __name__ == "__main__":
    query = "护眼台灯的库存是多少"
    doc_path = "~/Downloads/ryosuke_dev_qa_3.csv"
    document_index = "table_qa.keywords.ryosuke_dev_qa_3.v1"
    index_data_from_table(doc_path=doc_path, document_index=document_index)
    query_table(query=query, document_index=document_index)
