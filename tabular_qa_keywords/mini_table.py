import logging

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes.retriever import BM25Retriever
from haystack import Document

import pandas as pd

from tabular_qa_semantic_search import preprocess


class MiniTableQA:
    document_index: str
    document_store: ElasticsearchDocumentStore
    retriever: BM25Retriever

    def __init__(self, document_index: str):
        self.setupLogging()

        self.document_index = document_index

        logging.getLogger("haystack").info(
            "🚧 start to connect to Elasticsearch")
        self.document_store = ElasticsearchDocumentStore(
            username="admin", password="admin", index=document_index)
        logging.getLogger("haystack").info(
            "✅ succeeds to connect to Elasticsearch")

        self.retriever = BM25Retriever(document_store=self.document_store)

    def setupLogging(self) -> None:
        logging.basicConfig(
            format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
        logging.getLogger("haystack").setLevel(logging.INFO)

    def index_data_from_table(self, doc_path: str) -> None:

        # indexing data
        logging.getLogger("haystack").info(
            "🚧 start to read table as documents")
        df = pd.read_csv(doc_path, index_col=0)
        df.fillna(value="", inplace=True)  # Minimal cleaning
        questions_and_answers = preprocess.generate_standardize_qa_from_table(
            table=df, question_template="$index_label, $column_label", columns=["question1", "answer"])
        questions_and_answers["answer"] = questions_and_answers["answer"].apply(
            lambda x: str(x))
        records = questions_and_answers.to_dict(orient="records")
        documents = [Document.from_dict(
            dict=r, field_map={"question1": "content"}) for r in records]
        logging.getLogger("haystack").info(
            "✅ succeeds to read table as documents")

        for d in documents:
            logging.getLogger("haystack").debug(d.to_dict())

        # add the table to the DocumentSto
        logging.getLogger("haystack").info(
            "🚧 start to write documents to Elasticsearch")
        self.document_store.write_documents(
            documents, index=self.document_index)
        logging.getLogger("haystack").info(
            "✅ succeeds to write documents to Elasticsearch")

    def query(self, query: str) -> None:
        logging.getLogger("haystack").info(
            "🚧 start to retrieve relavant tables with Retriever")
        retrived_docs = self.retriever.retrieve(
            query, index=self.document_index, top_k=5)
        logging.getLogger("haystack").info(
            "✅ succeeds to retrieve relavant tables with Retriever")
        for doc in retrived_docs:
            print(doc)

    def evaluate_table(self, doc_path: str) -> pd.DataFrame:
        logging.getLogger("haystack").info(
            "🚧 start to write evaluation data")

        input_table = pd.read_csv(doc_path, index_col=0)
        std_qa_table = preprocess.generate_standardize_qa_from_table(
            table=input_table, question_template="$index_label的$column_label是什么？", columns=["问题", "预期回答"])
        result = []
        for content in std_qa_table.itertuples():
            question = content[1]
            expected_answer = content[2]
            retrived_doc = self.retriever.retrieve(
                query=question, index=self.document_index, top_k=1)[0]
            actual_answer = self.answer_from_retrieved_doc(retrived_doc)
            answer_context = self.context_from_retrieved_doc(retrived_doc)

            logging.getLogger("haystack").info(
                f"✅ evalution item created, question: {question}, expected answer: {expected_answer}, actual answer: {actual_answer}, answer context: {answer_context}")

            result.append([
                question, expected_answer, actual_answer, answer_context
            ])

        logging.getLogger("haystack").info(
            "✅ finish writting evaluation data")

        return pd.DataFrame(result, columns=["问题", "预期回答", "实际回答", "回答上下文"])

    def answer_from_retrieved_doc(self, doc: Document) -> str:
        return doc.meta.get("answer", "")

    def context_from_retrieved_doc(self, table: Document) -> str:
        if isinstance(table.content, str):
            return table.content
        else:
            return ""

if __name__ == "__main__":
    doc_path = "~/Downloads/ryosuke_dev_qa_3.csv"
    document_index = "table_qa_keywords.minitable.ryosuke_dev_qa_3.v1"
    query = "护眼台灯的库存是多少"

    qa = MiniTableQA(document_index=document_index)
    # qa.index_data_from_table(doc_path=doc_path)
    # qa.query(query)
    evaluation_result = qa.evaluate_table(
        doc_path=doc_path)
    evaluation_result.to_csv("~/Desktop/tabular_qa_minitable.eval.ryosuke_dev_qa_3.csv")
