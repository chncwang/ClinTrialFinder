import unittest
from base.disease_expert import is_oncology_disease

class TestIsOncologyDisease(unittest.TestCase):
    def test_oncology_keywords(self):
        oncology_terms = [
            'lung cancer', 'breast carcinoma', 'sarcoma', 'acute leukemia',
            'lymphoma', 'brain tumor', 'malignant neoplasm', 'melanoma',
            'neuroblastoma', 'multiple myeloma', 'metastatic colon cancer'
        ]
        for term in oncology_terms:
            with self.subTest(term=term):
                self.assertTrue(is_oncology_disease(term))

    def test_non_oncology_keywords(self):
        non_oncology_terms = [
            'diabetes', 'hypertension', 'asthma', 'eczema', 'arthritis',
            'pneumonia', 'migraine', 'fracture', 'depression', 'anemia'
        ]
        for term in non_oncology_terms:
            with self.subTest(term=term):
                self.assertFalse(is_oncology_disease(term))

if __name__ == '__main__':
    unittest.main() 