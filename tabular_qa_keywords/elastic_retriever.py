import pandas as pd
from typing import List


def flat_table_cell_to_table(table: pd.DataFrame) -> List[pd.DataFrame]:
    # Iterate over the rows of the input DataFrame
    output = []
    for i in range(1, len(table)):
        for j in range(1, len(table.columns)):
            entity = table.iloc[0, 0]
            entity_value = table.iloc[i, 0]
            property_name = table.iloc[0, j]
            property_value = table.iloc[i, j]
            # Create a new DataFrame for the entity's property
            property_df = pd.DataFrame([
                [entity, property_name],
                [entity_value, property_value]
            ])
            # Append the new DataFrame to the output list
            output.append(property_df)
    return output
