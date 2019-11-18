from src.methods.log_parser import LogParser
from src.utils import print_items


class Iplom(LogParser):
    def __init__(self, log_file_path):
        super().__init__(log_file_path)
        self.count_partitions = {}
        self.position_partitions = {}
        self.bijection_partitions = {}

    def parse(self):
        """
        Update fields corresponding to the list of templates and their correspondences.
        """
        self.count_partitions = self._get_token_count_partitions()
        self.position_partitions = self._get_token_position_partitions()
        # Partition by search for bijection
        # Discover cluster templates

    def print_count_partition(self):
        """
        Print all log entries grouped by their respective token count.
        """
        for count in self.count_partitions:
            log_indices = self.count_partitions[count]
            log_entries = self._get_log_entries_from_indices(log_indices)
            print('Count: {}'.format(count))
            print_items(log_entries)

    def print_position_partitions(self):
        """
        Print all of the log entries grouped by count and least unique token.
        """
        for count in self.position_partitions:
            subpartitions = self.position_partitions[count]
            for token in subpartitions:
                print('Count: {} \t Unique Token: {}'.format(count, token))
                log_entries = self._get_log_entries_from_indices(subpartitions[token])
                print_items(log_entries)

    def _get_log_entries_from_indices(self, log_indices):
        return [' '.join(tokenized_log_entry) for tokenized_log_entry in
                self._get_tokenized_log_entries_from_indices(log_indices)]

    def _get_tokenized_log_entries_from_indices(self, log_indices):
        return [self.tokenized_log_entries[log_id] for log_id in log_indices]

    def _get_token_count_partitions(self):
        """
        Returns a dict where the key is a count and the value is a
        list of log entry indices.
        """
        count_partitions = {}
        for idx, log_entry in enumerate(self.tokenized_log_entries):
            n = len(log_entry)
            if n not in count_partitions:
                count_partitions[n] = []
            count_partitions[n].append(idx)
        return count_partitions

    def _get_token_position_partitions(self):
        """
        Returns a nested dict where the two keys corresponds to the count and
        token position while the value is a list of log entry indices.
        """
        total_position_partitions = {count: {} for count in self.count_partitions}
        for count in self.count_partitions:
            log_indices = self.count_partitions[count]
            least_unique_token_index = self._get_least_unique_token_index(log_indices)
            total_position_partitions[count] = \
                self._get_positions_partitions(least_unique_token_index, log_indices)
        return total_position_partitions

    def _get_positions_partitions(self, least_unique_token_index, log_indices):
        """
        Returns a dict where the key corresponds to the token while
        the value is a list of log entry indices.
        """
        positions_partitions = {}
        for log_idx in log_indices:
            log_entry = self.tokenized_log_entries[log_idx]
            token = log_entry[least_unique_token_index]
            if token not in positions_partitions:
                positions_partitions[token] = []
            positions_partitions[token].append(log_idx)
        return positions_partitions

    def _get_least_unique_token_index(self, log_indices):
        """
        Returns: (1) the index corresponding to the token position with the least
        number of unique tokens as well as (2) the set of corresponding tokens.
        """
        self._validate_tokenized_log_entries_length(log_indices)
        unique_tokens = self._get_unique_tokens(log_indices)
        least_unique_col_idx = None
        least_unique_col_count = None
        for idx in unique_tokens:
            count = len(unique_tokens[idx])
            if (least_unique_col_idx is None) or (count < least_unique_col_count):
                least_unique_col_idx = idx
                least_unique_col_count = count
        return least_unique_col_idx

    def _get_unique_tokens(self, tokenized_log_indices):
        """
        Returns a dict where the key is the token position and the value is a set of unique tokens.
        """
        unique_tokens = {}
        for log_idx in tokenized_log_indices:
            log_entry = self.tokenized_log_entries[log_idx]
            for token_idx in range(len(log_entry)):
                if token_idx not in unique_tokens:
                    unique_tokens[token_idx] = set()
                unique_tokens[token_idx].add(log_entry[token_idx])
        return unique_tokens

    def _validate_tokenized_log_entries_length(self, tokenized_log_indices):
        """
        Check if logs all have the same token length.
        """
        for log_idx in tokenized_log_indices:
            ref_log_idx = tokenized_log_indices[0]
            expected_length = len(self.tokenized_log_entries[ref_log_idx])
            actual_length = len(self.tokenized_log_entries[log_idx])
            if expected_length != actual_length:
                error_message = 'IPLoM: Invalid log entry length ---> Expected: {} / Actual: {}'
                raise Exception(error_message.format(expected_length, actual_length))

    def _partition_by_bijection_search(self):
        pass

    def _extract_templates(self):
        pass

    def get_templates(self):
        return self.unique_event_templates
