import pandas as pd

from typing import List

import logging

from haystack import Document

import preprocess

from haystack.document_stores import ElasticsearchDocumentStore

from haystack.nodes.retriever import EmbeddingRetriever


class MiniTableQA:
    document_store: ElasticsearchDocumentStore
    retriever: EmbeddingRetriever

    def __init__(self):
        self.setupLogging()

        logging.getLogger("haystack").info(
            "🚧 start to connect to Elasticsearch")
        self.document_store = ElasticsearchDocumentStore(
            username="admin", password="admin", index=document_index)
        logging.getLogger("haystack").info(
            "✅ succeeds to connect to Elasticsearch")

        self.retriever = EmbeddingRetriever(
            document_store=self.document_store, embedding_model="deepset/all-mpnet-base-v2-table")

    def setupLogging(self) -> None:
        logging.basicConfig(
            format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
        logging.getLogger("haystack").setLevel(logging.INFO)

    def index_data_from_table(self, doc_path: str, document_index: str) -> None:

        # indexing data
        logging.getLogger("haystack").info(
            "🚧 start to read table as documents")
        df = pd.read_csv(doc_path, index_col=0)
        df.fillna(value="", inplace=True)  # Minimal cleaning
        tables = preprocess.flat_table_cell_to_table(table=df)
        documents = [Document(content=table, content_type="table")
                     for table in tables]
        logging.getLogger("haystack").info(
            "✅ succeeds to read table as documents")

        # add the table to the DocumentStore
        logging.getLogger("haystack").info(
            "🚧 start to write documents to Elasticsearch")
        self.document_store.write_documents(documents, index=document_index)
        logging.getLogger("haystack").info(
            "✅ succeeds to write documents to Elasticsearch")

        # update embeddings
        logging.getLogger("haystack").info("🚧 start to update embeddings")
        self.document_store.update_embeddings(retriever=self.retriever)
        logging.getLogger("haystack").info("✅ succeeds to update embeddings")

    def query_table(self, query: str, document_index: str) -> None:
        logging.getLogger("haystack").info(
            "🚧 start to retrieve relavant tables with Retriever")
        retrived_tables = self.retriever.retrieve(
            query, index=document_index, top_k=5)
        logging.getLogger("haystack").info(
            "✅ succeeds to retrieve relavant tables with Retriever")
        for table in retrived_tables:
            print(table)


if __name__ == "__main__":
    query = "护眼台灯的库存是多少"
    doc_path = "~/Downloads/ryosuke_dev_qa_3.csv"
    document_index = "table_qa.keywords.ryosuke_dev_qa_3.v1"
    qa = MiniTableQA()
    qa.index_data_from_table(doc_path=doc_path, document_index=document_index)
    qa.query_table(query=query, document_index=document_index)
