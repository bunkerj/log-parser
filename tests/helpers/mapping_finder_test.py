import unittest
from src.helpers.mapping_finder import MappingFinder

p1_token_mapping_1 = {
    'A': {'I'},
    'B': {'J', 'K'},
    'C': {'L'},
    'D': {'L'},
    'E': {'M', 'N'},
    'F': {'N', 'O', 'P'},
    'G': {'P'},
    'H': {'P'},
}

p2_token_mapping_1 = {
    'I': {'A'},
    'J': {'B'},
    'K': {'B'},
    'L': {'C', 'D'},
    'M': {'E'},
    'N': {'E', 'F'},
    'O': {'F'},
    'P': {'F', 'G', 'H'},
}

expected_results_1 = {
    'A': [{'A'}, {'I'}],
    'B': [{'B'}, {'J', 'K'}],
    'C': [{'C', 'D'}, {'L'}],
    'D': [{'C', 'D'}, {'L'}],
    'E': [{'E', 'F', 'G', 'H'}, {'M', 'N', 'O', 'P'}],
    'F': [{'E', 'F', 'G', 'H'}, {'M', 'N', 'O', 'P'}],
    'G': [{'E', 'F', 'G', 'H'}, {'M', 'N', 'O', 'P'}],
    'H': [{'E', 'F', 'G', 'H'}, {'M', 'N', 'O', 'P'}],
}

p1_token_mapping_2 = {
    'A': {'F', 'G'},
    'B': {'G', 'F'},
}

p2_token_mapping_2 = {
    'F': {'A', 'B'},
    'G': {'B', 'A'},
}

expected_results_2 = {
    'A': [{'A', 'B'}, {'F', 'G'}],
    'B': [{'A', 'B'}, {'F', 'G'}],
    'F': [{'A', 'B'}, {'F', 'G'}],
    'G': [{'A', 'B'}, {'F', 'G'}],
}


class TestMappingFinder(unittest.TestCase):
    def test_A(self):
        self._test(p1_token_mapping_1, p2_token_mapping_1, expected_results_1)

    def test_B(self):
        self._test(p1_token_mapping_2, p2_token_mapping_2, expected_results_2)

    def _test(self, p1_token_mapping, p2_token_mapping, expected_results):
        mapping_finder = MappingFinder(p1_token_mapping, p2_token_mapping)
        for token in p1_token_mapping:
            mapping_finder.update_relevant_token_sets(token)
            domain_set = mapping_finder.domain_set
            codomain_set = mapping_finder.codomain_set

            domain_set_expected, codomain_set_expected = expected_results[token]
            self.assertEqual(domain_set, domain_set_expected)
            self.assertEqual(codomain_set, codomain_set_expected)


if __name__ == '__main__':
    unittest.main()
