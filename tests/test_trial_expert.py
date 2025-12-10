#!/usr/bin/env python3
import sys
from pathlib import Path
import unittest

# Add parent directory to Python path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from base.disease_expert import is_oncology_disease
from base.trial_expert import deduplicate_drug_names


class TestDeduplicateDrugNames(unittest.TestCase):
    def test_empty_list(self):
        """Test that empty list returns empty list."""
        result = deduplicate_drug_names([])
        self.assertEqual(result, [])

    def test_no_duplicates(self):
        """Test that list with no duplicates remains unchanged."""
        drugs = ["Nivolumab", "Pembrolizumab", "Ipilimumab"]
        result = deduplicate_drug_names(drugs)
        self.assertEqual(result, drugs)

    def test_exact_duplicates(self):
        """Test that exact duplicate names are removed."""
        drugs = ["Nivolumab", "Pembrolizumab", "Nivolumab"]
        result = deduplicate_drug_names(drugs)
        self.assertEqual(result, ["Nivolumab", "Pembrolizumab"])

    def test_case_insensitive_duplicates(self):
        """Test that case variations are treated as duplicates."""
        drugs = ["Nivolumab", "nivolumab", "NIVOLUMAB"]
        result = deduplicate_drug_names(drugs)
        # Should keep first occurrence with original casing
        self.assertEqual(result, ["Nivolumab"])
        self.assertEqual(len(result), 1)

    def test_mixed_case_duplicates(self):
        """Test mixed case variations with other drugs."""
        drugs = ["BMS-936558", "bms-936558", "Pembrolizumab", "pembrolizumab"]
        result = deduplicate_drug_names(drugs)
        self.assertEqual(result, ["BMS-936558", "Pembrolizumab"])

    def test_whitespace_handling(self):
        """Test that whitespace is properly stripped during comparison."""
        drugs = ["Nivolumab", " Nivolumab ", "  nivolumab"]
        result = deduplicate_drug_names(drugs)
        self.assertEqual(len(result), 1)

    def test_preserves_first_occurrence_casing(self):
        """Test that the first occurrence's casing is preserved."""
        drugs = ["nivolumab", "Nivolumab", "NIVOLUMAB"]
        result = deduplicate_drug_names(drugs)
        self.assertEqual(result, ["nivolumab"])

    def test_multiple_drug_combinations(self):
        """Test realistic scenario with multiple drugs and duplicates."""
        drugs = [
            "BMS-936558",
            "Nivolumab",
            "bms-936558",  # Duplicate of first
            "Pembrolizumab",
            "nivolumab",    # Duplicate of second
            "Ipilimumab"
        ]
        result = deduplicate_drug_names(drugs)
        self.assertEqual(result, ["BMS-936558", "Nivolumab", "Pembrolizumab", "Ipilimumab"])


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