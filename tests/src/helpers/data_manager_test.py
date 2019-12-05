import re
import unittest
from src.helpers.data_manager import DataManager

data_manager = DataManager({})


class TestMappingFinder(unittest.TestCase):
    def test_simple_template(self):
        template = 'total of <*> ddr error(s) detected and corrected over <*> seconds'
        regex = data_manager._get_template_regex(template)

        match = re.match(regex, 'total of 20 ddr error(s) detected and corrected over 13 seconds')
        self.assertIsNotNone(match)
        match = re.match(regex, 'total of 20  ddr error(s) detected and corrected over 13 seconds')
        self.assertIsNotNone(match)
        match = re.match(regex, 'total of  20 ddr error(s) detected and corrected over 13 seconds')
        self.assertIsNotNone(match)

        match = re.match(regex, 'total of   20 ddr error(s) detected and corrected over 13 seconds')
        self.assertIsNone(match)
        match = re.match(regex, 'total of 20   ddr error(s) detected and corrected over 13 seconds')
        self.assertIsNone(match)
        match = re.match(regex, 'total of 20ddr error(s) detected and corrected over 13 seconds')
        self.assertIsNone(match)
        match = re.match(regex, 'total of ddr error(s) detected and corrected over seconds')
        self.assertIsNone(match)
        match = re.match(regex, 'total of 20 ddr error(s) detected and corrected over 13 seconds,')
        self.assertIsNone(match)

    def test_sequence_of_wildcards(self):
        template = 'wanted: <*> <*> <*> <*> <*> <*> <*> <*> got: <*> <*> <*> <*> <*> <*>'
        regex = data_manager._get_template_regex(template)

        match = re.match(regex, 'wanted: A C X+ X- Y+ Y- Z+ Z- got: A C X+ Y- Z+ Z-')
        self.assertIsNotNone(match)
        match = re.match(regex, 'wanted: A C X+ X- Y+ Y- Z+ Z- got: A C X+ Y- Z+ ')
        self.assertIsNotNone(match)
        match = re.match(regex, 'wanted: A C X+ X- Y+ Y- Z+ Z- got: A C X+ Y- Z+')
        self.assertIsNotNone(match)

        match = re.match(regex, 'wanted: A C X+ X- Y+ Y- Z+ Z- got: A C X+ Y-')
        self.assertIsNone(match)
        match = re.match(regex, 'wanted: C X+ X- Y+ Y- Z+ Z- got: A C X+ Y- Z+ Z-')
        self.assertIsNone(match)

    def test_wildcard_suffix(self):
        template = 'generating <*>'
        regex = data_manager._get_template_regex(template)

        match = re.match(regex, 'generating 0387')
        self.assertIsNotNone(match)
        match = re.match(regex, 'generating ')
        self.assertIsNotNone(match)
        match = re.match(regex, 'generating')
        self.assertIsNotNone(match)

        match = re.match(regex, 'generating 0387 012')
        self.assertIsNone(match)
        match = re.match(regex, ' generating 0387')
        self.assertIsNone(match)

    def test_wildcard_prefix(self):
        template = '<*> <*> failed to lock'
        regex = data_manager._get_template_regex(template)

        match = re.match(regex, 'A B failed to lock')
        self.assertIsNotNone(match)
        match = re.match(regex, ' A B failed to lock')
        self.assertIsNotNone(match)

        match = re.match(regex, '  A B failed to lock')
        self.assertIsNone(match)
        match = re.match(regex, 'B failed to lock')
        self.assertIsNone(match)

    def test_no_space_wildcards_1(self):
        template = 'lr:<*> cr:<*> xer:<*> ctr:<*>'
        regex = data_manager._get_template_regex(template)

        match = re.match(regex, 'lr:12 cr:132 xer:342 ctr:9383')
        self.assertIsNotNone(match)

        match = re.match(regex, 'lr:12cr:132 xer:342 ctr:9383')
        self.assertIsNone(match)
        match = re.match(regex, 'lr:12  cr:132   xer:342  ctr:9383')
        self.assertIsNone(match)

    def test_no_space_wildcards_2(self):
        template = 'ciod: LOGIN chdir(<*>) failed: Input/output error'
        regex = data_manager._get_template_regex(template)

        match = re.match(regex, 'ciod: LOGIN chdir(123) failed: Input/output error')
        self.assertIsNotNone(match)
        match = re.match(regex, 'ciod: LOGIN chdir( 123) failed: Input/output error')
        self.assertIsNotNone(match)
        match = re.match(regex, 'ciod: LOGIN chdir(123 ) failed: Input/output error')
        self.assertIsNotNone(match)
        match = re.match(regex, 'ciod: LOGIN chdir( 123 ) failed: Input/output error')
        self.assertIsNotNone(match)
        match = re.match(regex, 'ciod: LOGIN chdir() failed: Input/output error')
        self.assertIsNotNone(match)
        match = re.match(regex, 'ciod: LOGIN chdir( ) failed: Input/output error')
        self.assertIsNotNone(match)
        match = re.match(regex, 'ciod: LOGIN chdir(  ) failed: Input/output error')
        self.assertIsNotNone(match)

        match = re.match(regex, 'ciod: LOGIN chdir(123  ) failed: Input/output error')
        self.assertIsNone(match)
        match = re.match(regex, 'ciod: LOGIN chdir(  123 ) failed: Input/output error')
        self.assertIsNone(match)
        match = re.match(regex, 'ciod: LOGIN chdir(   ) failed: Input/output error')
        self.assertIsNone(match)

    def test_single_wildcard(self):
        template = '<*>'
        regex = data_manager._get_template_regex(template)

        match = re.match(regex, 'test')
        self.assertIsNotNone(match)
        match = re.match(regex, '')
        self.assertIsNotNone(match)
        match = re.match(regex, ' ')
        self.assertIsNotNone(match)
        match = re.match(regex, '  ')
        self.assertIsNotNone(match)

        match = re.match(regex, '   ')
        self.assertIsNone(match)

    def test_no_wildcards(self):
        template = 'program interrupt'
        regex = data_manager._get_template_regex(template)

        match = re.match(regex, 'program interrupt')
        self.assertIsNotNone(match)

        match = re.match(regex, ' program interrupt')
        self.assertIsNone(match)
        match = re.match(regex, 'program interrupt ')
        self.assertIsNone(match)
        match = re.match(regex, 'program  interrupt')
        self.assertIsNone(match)
        match = re.match(regex, 'pr0gram interrupt')
        self.assertIsNone(match)


if __name__ == '__main__':
    unittest.main()
