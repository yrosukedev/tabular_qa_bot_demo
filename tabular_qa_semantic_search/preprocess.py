import pandas as pd

from typing import List


def flat_table_cell_to_table(table: pd.DataFrame) -> List[pd.DataFrame]:
    # Iterate over the rows of the input DataFrame
    output = list()
    for column_label, content in table.items():
        for index_label, value in content.items():
            output.append(
                pd.DataFrame([index_label, value], columns=[column_label]))
    return output
