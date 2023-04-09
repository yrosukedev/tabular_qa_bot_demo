import unittest
import mini_table
import pandas as pd
from typing import List, TypeVar, Callable, Union


T = TypeVar('T')


def check_list_equal(list1: List[T], list2: List[T], matcher: Callable[[T, T], bool]) -> bool:
    for x, y in zip(list1, list2):
        if not matcher(x, y):
            return False
    return True


def data_frame_matcher(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    return df1.equals(df2) and df1.index.equals(df2.index) and df1.columns.equals(df2.columns)


class TestFlatTableCell(unittest.TestCase):

    def assertDataFrameListEqual(self, xs1: List[pd.DataFrame], xs2: List[pd.DataFrame], msg: Union[str, None] = None) -> None:
        self.assertTrue(check_list_equal(xs1, xs2, data_frame_matcher), msg)

    def assertDataFrameListNotEqual(self, xs1: List[pd.DataFrame], xs2: List[pd.DataFrame], msg: Union[str, None] = None) -> None:
        self.assertFalse(check_list_equal(xs1, xs2, data_frame_matcher), msg)

    def test_pandas_basic_usage(self):
        input = pd.DataFrame([
            ["Fruit", "Price", "Color"],
            ["Apple", "10$/KG", "Green"],
            ["Grep", "5$/KG", "purple"]
        ])
        print(f"index: {input.index}")
        print(f"columns: {input.columns}")
        print(f"values: {input.values}")
        print(f"axes: {input.axes}")
        print(f"ndim: {input.ndim}")
        print(f"size: {input.size}")
        print(f"shape: {input.shape}")

    def test_equality_of_two_data_frames(self):
        x = pd.DataFrame([
            ["10$/KG", "Green"],
            ["5$/KG", "purple"]
        ], columns=["Price", "Color"], index=["Apple", "Grep"])
        y = pd.DataFrame([
            ["10$/KG", "Green"],
            ["5$/KG", "purple"]
        ], columns=["Price", "Color"], index=["Apple", "Grep"])
        z = pd.DataFrame([
            ["10$/KG", "Green"],
            ["5$/KG", "purple"]
        ], columns=["Price", "Color X"], index=["Apple", "Grep"])

        self.assertDataFrameListEqual(
            [x, y], [x, y], "data frame lists should be equal")
        self.assertDataFrameListNotEqual(
            [x, y], [x, z], "data frame lists shouldn't be equal")

    def test_flat_table_cell_to_table(self):
        input = pd.DataFrame([
            ["10$/KG", "green"],
            ["5$/KG", "purple"]
        ], columns=["Price", "Color"], index=["Apple", "Grep"])

        expected_outputs = [
            pd.DataFrame(["Apple", "10$/KG"], columns=["Price"]),
            pd.DataFrame(["Grep", "5$/KG"], columns=["Price"]),
            pd.DataFrame(["Apple", "green"], columns=["Color"]),
            pd.DataFrame(["Grep", "purple"], columns=["Color"]),
        ]
        actual_outputs = mini_table.flat_table_cell_to_table(input)
        self.assertDataFrameListEqual(
            expected_outputs, actual_outputs, msg="data frame list should be equal")

    # def test_flat_real_table(self):
    #     doc_path = "~/Downloads/ryosuke_dev_qa_1.csv"
    #     input_table = pd.read_csv(doc_path, index_col=0)
    #     print(f"1st cell from the input table: \n{input_table.head(1)}")
    #     output_tables = elastic_retriever.flat_table_cell_to_table(input_table)
    #     print(f"1st output table: \n{output_tables[0]}")


if __name__ == "__main__":
    unittest.main()
