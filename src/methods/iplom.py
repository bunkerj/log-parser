from src.methods.log_parser import LogParser


class Iplom(LogParser):
    def __init__(self, log_file_path):
        super().__init__(log_file_path)

    def parse(self):
        """
        Update fields corresponding to the list of templates and their correspondences.
        """
        token_count = self._partition_by_token_count(self.tokenized_log_entries)
        result = self._partition_by_token_position(token_count)
        # Partition by search for bijection
        # Discover cluster templates
        return token_count

    def _partition_by_token_count(self, tokenized_log_entries):
        """
        Returns a dict where the key is a count and the value is a
        list of log entry indices.
        """
        token_count = {}
        for idx, log_entry in enumerate(tokenized_log_entries):
            n = len(log_entry)
            if n not in token_count:
                token_count[n] = []
            token_count[n].append(idx)
        return token_count

    def _partition_by_token_position(self, token_count):
        """
        Returns a dict where the key is a count and value is a dict.
        This child dict uses a key corresponding to the a token and
        the value is a list of log entry indices.
        """
        for n in token_count:
            tokenized_log_indices = token_count[n]
            col_idx, tokens = self._get_least_unique_col_index_and_tokens(tokenized_log_indices)
            print(tokens)
        return []

    def _get_least_unique_col_index_and_tokens(self, tokenized_log_indices):
        """
        Returns: (1) the index corresponding to the token position with the least
        number of unique tokens as well as (2) the set of corresponding tokens.
        """
        self._validate_tokenized_log_entries_length(tokenized_log_indices)
        unique_tokens = self._get_unique_tokens(tokenized_log_indices)
        least_unique_col_idx = None
        least_unique_col_count = None
        for idx in unique_tokens:
            n = len(unique_tokens[idx])
            if (least_unique_col_idx is None) or (n < least_unique_col_count):
                least_unique_col_idx = idx
                least_unique_col_count = n
        return least_unique_col_idx, unique_tokens[least_unique_col_idx]

    def _get_unique_tokens(self, tokenized_log_indices):
        """
        Returns a dict where the key is the idx and the value is a set of unique tokens.
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
