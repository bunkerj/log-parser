from src.helpers.partition_item import ParititionItem


class Partitions:
    def __init__(self, tokenized_log_entries):
        self.partition_list = []
        self.tokenized_log_entries = tokenized_log_entries

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self.partition_list):
            result = self.partition_list[self.idx]
            self.idx += 1
            return result
        else:
            raise StopIteration

    def add(self, log_indices, step_num):
        self._validate_partition_size(log_indices)
        self._validate_log_entry_lengths(log_indices)
        partition_item = ParititionItem(log_indices, step_num)
        self.partition_list.append(partition_item)

    def _validate_partition_size(self, log_indices):
        if len(log_indices) == 0:
            raise Exception('Partition cannot be empty')

    def _validate_log_entry_lengths(self, log_indices):
        ref_token_idx = log_indices[0]
        ref_token_length = len(self.tokenized_log_entries[ref_token_idx])
        for token_idx in log_indices:
            token_length = len(self.tokenized_log_entries[token_idx])
            if token_length != ref_token_length:
                error_msg = 'Invalid log entry length ---> Expected: {} / Actual: {}'
                raise Exception(error_msg.format(ref_token_length, token_length))
