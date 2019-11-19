import unittest
from src.helpers.mapping_finder import MappingFinder

p1_token_mapping = {
    'A': {'I'},
    'B': {'J', 'K'},
    'C': {'L'},
    'D': {'L'},
    'E': {'M', 'N'},
    'F': {'N', 'O', 'P'},
    'G': {'P'},
    'H': {'P'},
}

p2_token_mapping = {
    'I': {'A'},
    'J': {'B'},
    'K': {'B'},
    'L': {'C', 'D'},
    'M': {'E'},
    'N': {'E', 'F'},
    'O': {'F'},
    'P': {'F', 'G', 'H'},
}

expected_results = {
    'A': [{'A'}, {'I'}],
    'B': [{'B'}, {'J', 'K'}],
    'C': [{'C', 'D'}, {'L'}],
    'D': [{'C', 'D'}, {'L'}],
    'E': [{'E', 'F', 'G', 'H'}, {'M', 'N', 'O', 'P'}],
    'F': [{'E', 'F', 'G', 'H'}, {'M', 'N', 'O', 'P'}],
    'G': [{'E', 'F', 'G', 'H'}, {'M', 'N', 'O', 'P'}],
    'H': [{'E', 'F', 'G', 'H'}, {'M', 'N', 'O', 'P'}],
}


class TestMappingFinder(unittest.TestCase):

    def test_A(self):
        for token in p1_token_mapping:
            mapping_finder = MappingFinder(p1_token_mapping, p2_token_mapping)
            mapping_finder.update_relevant_token_sets(token)
            domain_set = mapping_finder.domain_set
            codomain_set = mapping_finder.codomain_set
            domain_set_expected, codomain_set_expected = expected_results[token]
            self.assertEqual(domain_set, domain_set_expected)
            self.assertEqual(codomain_set, codomain_set_expected)


if __name__ == '__main__':
    unittest.main()
