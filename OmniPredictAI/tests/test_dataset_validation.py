import os
import tempfile
import unittest

import pandas as pd

from app import answer_row_lookup_question, validate_dataset_file


class DatasetValidationTests(unittest.TestCase):
    def test_row_lookup_returns_requested_column_only(self):
        dataframe = pd.DataFrame([
            {'name': 'Ada Lovelace', 'address': '12 Analytical Engine Way', 'city': 'London'},
            {'name': 'Grace Hopper', 'address': '1 Navy Street', 'city': 'Arlington'},
        ])

        result = answer_row_lookup_question('What is the address of Ada Lovelace?', dataframe)

        self.assertEqual(result['message'], 'address: 12 Analytical Engine Way')

    def test_row_lookup_returns_entire_record(self):
        dataframe = pd.DataFrame([
            {'name': 'Ada Lovelace', 'address': '12 Analytical Engine Way', 'city': 'London'},
        ])

        result = answer_row_lookup_question('Tell me about Ada Lovelace', dataframe)

        self.assertIn('12 Analytical Engine Way', result['message'])
        self.assertIn('London', result['message'])

    def test_rejects_empty_or_too_small_dataset(self):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as handle:
            handle.write(b'col_a,col_b\n1,2\n')
            temp_path = handle.name

        try:
            with self.assertRaises(ValueError):
                validate_dataset_file(temp_path)
        finally:
            os.remove(temp_path)

    def test_accepts_valid_dataset(self):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as handle:
            pd.DataFrame({'age': [20, 25, 30], 'target': [0, 1, 1]}).to_csv(handle.name, index=False)
            temp_path = handle.name

        try:
            df = validate_dataset_file(temp_path)
            self.assertEqual(df.shape[0], 3)
            self.assertEqual(list(df.columns), ['age', 'target'])
        finally:
            os.remove(temp_path)

    def test_dont_have_missing_column_query(self):
        from app import answer_full_dataset_question
        dataframe = pd.DataFrame([
            {'Student Name': 'Alice', 'Blood Group': 'A+'},
            {'Student Name': 'Bob', 'Blood Group': None},
            {'Student Name': 'Charlie', 'Blood Group': ''},
        ])

        result = answer_full_dataset_question('tell me how many students dont have blood group', dataframe)
        self.assertIsNotNone(result)
        self.assertIn("2 (66.7%) do not have 'Blood Group' specified", result['message'])
        self.assertNotIn('**', result['message'])


if __name__ == '__main__':
    unittest.main()
