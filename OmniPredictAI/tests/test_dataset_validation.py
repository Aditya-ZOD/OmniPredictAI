import os
import tempfile
import unittest

import pandas as pd

from app import validate_dataset_file


class DatasetValidationTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
