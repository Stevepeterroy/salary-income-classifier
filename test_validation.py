import unittest
import pandas as pd
from validation import validate_batch
class InputTests(unittest.TestCase):
    def test_valid_and_unknown_category(self):
        self.assertEqual(len(validate_batch(pd.DataFrame([{"age":30,"hours-per-week":40,"education":"Unseen","occupation":"New"}]))),1)
    def test_invalid(self):
        base={"age":30,"hours-per-week":40,"education":"Bachelors","occupation":"Sales"}
        for change in [{"age":17},{"age":float("inf")},{"age":30.5},{"hours-per-week":0},{"education":None},{"occupation":""}]:
            with self.subTest(change=change),self.assertRaises(ValueError): validate_batch(pd.DataFrame([{**base,**change}]))
        with self.assertRaises(ValueError): validate_batch(pd.DataFrame())
