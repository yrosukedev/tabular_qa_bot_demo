import pandas as pd

from typing import List

from string import Template


def flat_table_cell_to_table(table: pd.DataFrame) -> List[pd.DataFrame]:
    # Iterate over the rows of the input DataFrame
    output = list()
    for column_label, content in table.items():
        for index_label, value in content.items():
            output.append(
                pd.DataFrame([index_label, value], columns=[column_label]))
    return output


def generate_standardize_qa_from_table(table: pd.DataFrame, question_template: str, columns: List[str]) -> pd.DataFrame:
    qaList = []
    for column_label, content in table.items():
        for index_label, answer in content.items():
            question = Template(template=question_template).substitute(
                index_label=index_label, column_label=column_label)
            qaList.append(
                [question, answer]
            )
    result = pd.DataFrame(qaList, columns=columns)
    return result
