from src.methods.log_parser import LogParser
from src.utils import print_items, get_n_sorted
from src.helpers.mapping_finder import MappingFinder
from src.constants import MAP
from copy import deepcopy


class Iplom(LogParser):
    def __init__(self, log_file_path, lower_bound, upper_bound):
        super().__init__(log_file_path)
        self.count_partitions = {}
        self.position_partitions = {}
        self.bijection_partitions = {}
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def parse(self):
        """
        Update fields corresponding to the list of templates and their correspondences.
        """
        self.count_partitions = self._get_token_count_partitions()
        self.position_partitions = self._get_token_position_partitions()
        self.bijection_partitions = self._get_bijection_partitions()
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
            tokenized_log_entries = self._get_tokenized_log_entries_from_indices(log_indices)
            least_unique_token_index = self._get_least_unique_token_index(tokenized_log_entries)
            total_position_partitions[count] = \
                self._get_position_partitions(least_unique_token_index, log_indices)
        return total_position_partitions

    def _get_position_partitions(self, least_unique_token_index, log_indices):
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

    def _get_least_unique_token_index(self, tokenized_log_entries):
        """
        Returns: (1) the index corresponding to the token position with the least
        number of unique tokens as well as (2) the set of corresponding tokens.
        """
        self._validate_tokenized_log_entries_length(tokenized_log_entries)
        unique_tokens = self._get_unique_tokens(tokenized_log_entries)
        least_unique_col_idx = None
        least_unique_col_count = None
        for idx in unique_tokens:
            count = len(unique_tokens[idx])
            if (least_unique_col_idx is None) or (count < least_unique_col_count):
                least_unique_col_idx = idx
                least_unique_col_count = count
        return least_unique_col_idx

    def _get_unique_tokens(self, tokenized_log_entries):
        """
        Returns a dict where the key is the token position and the value is a set of unique tokens.
        """
        unique_tokens = {}
        for log_entry in tokenized_log_entries:
            for token_idx in range(len(log_entry)):
                if token_idx not in unique_tokens:
                    unique_tokens[token_idx] = set()
                unique_tokens[token_idx].add(log_entry[token_idx])
        return unique_tokens

    def _validate_tokenized_log_entries_length(self, tokenized_log_entries):
        """
        Check if logs all have the same token length.
        """
        for log_entry in tokenized_log_entries:
            ref_log_entry = tokenized_log_entries[0]
            expected_length = len(ref_log_entry)
            actual_length = len(log_entry)
            if expected_length != actual_length:
                error_message = 'IPLoM: Invalid log entry length ---> Expected: {} / Actual: {}'
                raise Exception(error_message.format(expected_length, actual_length))

    def _get_bijection_partitions(self):
        bijection_partitions = self._initialize_bijection_partitions()
        for count in self.position_partitions:
            for position_token in self.position_partitions[count]:
                partition_log_indices = deepcopy(self.position_partitions[count][position_token])
                p_in = deepcopy(self._get_tokenized_log_entries_from_indices(partition_log_indices))

                p1, p2 = self._determineP1andP2(p_in)
                tmp_partitions = {p1: {}, p2: {}}
                p1_token_mapping = self._get_token_mapping(p_in, p1, p2)
                p2_token_mapping = self._get_token_mapping(p_in, p2, p1)

                mapping_finder = MappingFinder(p1_token_mapping, p2_token_mapping)

                for token in p1_token_mapping:
                    mapping_finder.update_relevant_token_sets(token)
                    token_sets = {p1: mapping_finder.domain_set, p2: mapping_finder.codomain_set}
                    map_type = self._get_map_type(token_sets[p1], token_sets[p2])
                    if map_type == MAP.ONE_TO_ONE:
                        split_pos = p1
                    elif map_type == MAP.ONE_TO_MANY:
                        s_temp = token_sets[p2]
                        split_rank = self._get_rank_positions(p_in, p2, s_temp, True)
                        if split_rank == 1:
                            split_pos = p1
                        else:
                            split_pos = p2
                    elif map_type == MAP.MANY_TO_ONE:
                        s_temp = token_sets[p1]
                        split_rank = self._get_rank_positions(p_in, p2, s_temp, False)
                        if split_rank == 2:
                            split_pos = p2
                        else:
                            split_pos = p1
                    else:
                        if len(self.position_partitions[count]) > 1:
                            continue
                        else:
                            s_temp1 = token_sets[p1]
                            s_temp2 = token_sets[p2]
                            if len(s_temp1) < len(s_temp2):
                                split_pos = p1
                            else:
                                split_pos = p2

                    # Split into new partitions based on split_pos
                    for idx, tokenized_log_entry in enumerate(p_in):
                        if tokenized_log_entry[p1] == token:
                            split_token = tokenized_log_entry[split_pos]
                            if split_token not in tmp_partitions[split_pos]:
                                tmp_partitions[split_pos][split_token] = []
                            tmp_partitions[split_pos][split_token].append(partition_log_indices[idx])
                            partition_log_indices.pop(idx)
                            p_in.pop(idx)

                    if len(p_in) == 0:
                        break

                # Add new partitions in tmp_partitions to bijection_partitions
                partition_idx = 0
                for split_pos in tmp_partitions:
                    for split_token in tmp_partitions[split_pos]:
                        bijection_partitions[count][position_token][partition_idx] \
                            = tmp_partitions[split_pos][split_token]
                        partition_idx += 1

                # Create a new partition with the remaining lines (from M-M relationships)
                # TODO: Split disjoint M-M groups
                if len(p_in) > 0:
                    bijection_partitions[count][position_token][partition_idx] = partition_log_indices

        return bijection_partitions

    def _initialize_bijection_partitions(self):
        return {c: {p: {} for p in self.position_partitions[c]}
                for c in self.position_partitions}

    def _determineP1andP2(self, tokenized_log_entries):
        """
        Return the two most frequent token positions.
        """
        unique_tokens = self._get_unique_tokens(tokenized_log_entries)
        count_per_token_idx = [(len(unique_tokens[token_idx]), token_idx) for token_idx in unique_tokens]
        counts = get_n_sorted(2, count_per_token_idx, key=lambda x: x[0], get_max=True)
        return sorted(count[1] for count in counts)

    def _get_token_mapping(self, p_in, domain_idx, codomain_idx):
        """
        Return mapping between tokens in domain to tokens in codomain.
        """
        token_mapping = {}
        for log_entry in p_in:
            domain_token = log_entry[domain_idx]
            codomain_token = log_entry[codomain_idx]
            if domain_token not in token_mapping:
                token_mapping[domain_token] = set()
            token_mapping[domain_token].add(codomain_token)
        return token_mapping

    def _get_map_type(self, domain_token_set, codomain_token_set):
        """
        Returns the map type given the associated token sets for a particular token.
        """
        if len(domain_token_set) == len(codomain_token_set) == 1:
            return MAP.ONE_TO_ONE
        elif len(domain_token_set) == 1 and len(codomain_token_set) > 1:
            return MAP.ONE_TO_MANY
        elif len(domain_token_set) > 1 and len(codomain_token_set) == 1:
            return MAP.MANY_TO_ONE
        else:
            return MAP.MANY_TO_MANY

    def _get_rank_positions(self, p_in, token_idx, s_temp, is_one_to_m):
        line_count = 0
        for log_entry in p_in:
            if log_entry[token_idx] in s_temp:
                line_count += 1

        distance = len(s_temp) / line_count
        if distance < self.lower_bound:
            return 2 if is_one_to_m else 1
        else:
            return 1 if is_one_to_m else 2

    def _extract_templates(self):
        pass

    def get_templates(self):
        return self.unique_event_templates
