class DataCompressor:
    def __init__(self, tokenized_logs):
        self.tokenized_logs = tokenized_logs

    def get_coreset(self):
        unique_logs = {}
        occurrences = {}
        for log in self.tokenized_logs:
            key = ' '.join(log)
            if key not in unique_logs:
                unique_logs[key] = log
                occurrences[key] = 0
            occurrences[key] += 1

        reduced_weights = list(occurrences.values())
        reduced_set = list(unique_logs.values())

        return reduced_weights, reduced_set
