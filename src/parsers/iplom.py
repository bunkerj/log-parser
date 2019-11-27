from src.parsers.log_parser import LogParser
from src.utils import print_items, get_n_sorted, delete_indices_from_list
from src.helpers.mapping_finder import MappingFinder
from src.helpers.partitions import Partitions
from constants import MAP, PLACEHOLDER
from copy import deepcopy


class Iplom(LogParser):
    def __init__(self, data_config, file_threshold, partition_threshold,
                 lower_bound, upper_bound, goodness_threshold):
        super().__init__(data_config)
        self.partitions = Partitions(self.tokenized_log_entries)
        self.file_threshold = file_threshold
        self.partition_threshold = partition_threshold
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.goodness_threshold = goodness_threshold

    def parse(self):
        """
        Perform 3 partitioning steps followed by template generation.
        This function will update the partitions and cluster_templates fields.
        """
        self._partition_by_count()
        self._partition_by_position()
        self._partition_by_bijections()
        self._discover_cluster_templates()

    def print_partitions(self):
        for partition_item in self.partitions:
            log_entries = self._get_log_entries_from_indices(partition_item.log_indices)
            print_items(log_entries)

    def _get_log_entries_from_indices(self, log_indices):
        return [' '.join(tokenized_log_entry) for tokenized_log_entry in
                self._get_tokenized_log_entries_from_indices(log_indices)]

    def _get_tokenized_log_entries_from_indices(self, log_indices):
        return [self.tokenized_log_entries[log_id] for log_id in log_indices]

    def _partition_by_count(self):
        """
        Split partitions by token count.
        """
        count_dict = {}
        for idx, log_entry in enumerate(self.tokenized_log_entries):
            n = len(log_entry)
            if n not in count_dict:
                count_dict[n] = []
            count_dict[n].append(idx)

        count_partitions = Partitions(self.tokenized_log_entries)
        for count in count_dict:
            count_partitions.add(count_dict[count], 1)

        self.partitions = count_partitions

    def _partition_by_position(self):
        """
        Split partitions by the least unique token position.
        """
        position_partitions = Partitions(self.tokenized_log_entries)

        for partition_item in self.partitions:
            outlier_log_indices = []
            log_indices = partition_item.log_indices
            tokenized_log_entries = self._get_tokenized_log_entries_from_indices(log_indices)
            least_unique_token_index = self._get_least_unique_token_index(tokenized_log_entries)
            child_partitions = self._get_position_subpartitions_dict(least_unique_token_index, log_indices)
            for token in child_partitions:
                child_partition = child_partitions[token]
                partition_support = len(child_partition) / len(tokenized_log_entries)
                if partition_support < self.partition_threshold:
                    outlier_log_indices.extend(child_partition)
                else:
                    partition_step = 1 if len(child_partitions) == 1 else 2
                    position_partitions.add(child_partition, partition_step)

            if len(outlier_log_indices) > 0:
                position_partitions.add(outlier_log_indices, 2)

        self.partitions = position_partitions
        self._prune_partitions()

    def _get_position_subpartitions_dict(self, least_unique_token_index, log_indices):
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
        Returns a dict where the key is the token position
        and the value is a set of unique tokens.
        """
        unique_tokens = {}
        for log_entry in tokenized_log_entries:
            for token_idx in range(len(log_entry)):
                if token_idx not in unique_tokens:
                    unique_tokens[token_idx] = set()
                unique_tokens[token_idx].add(log_entry[token_idx])
        return unique_tokens

    def _partition_by_bijections(self):
        """
        Split partitions by seeking bijective relationships.
        """
        bijection_partitions = Partitions(self.tokenized_log_entries)
        for partition_item in self.partitions:
            log_indices = deepcopy(partition_item.log_indices)
            p_in = deepcopy(self._get_tokenized_log_entries_from_indices(log_indices))

            # Check for goodness threshold and token entry size
            partition_goodness = self._get_partition_goodness(p_in)
            if partition_goodness >= self.goodness_threshold or len(p_in[0]) < 2:
                bijection_partitions.add(log_indices, 2)
                continue

            p1, p2 = self._determine_p1_and_p2(p_in, partition_item.step)
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
                    split_rank = self._get_rank_positions(p_in, p1, s_temp, False)
                    if split_rank == 2:
                        split_pos = p2
                    else:
                        split_pos = p1
                else:
                    if partition_item.step == 2:
                        continue
                    else:
                        s_temp1 = token_sets[p1]
                        s_temp2 = token_sets[p2]
                        if len(s_temp1) < len(s_temp2):
                            split_pos = p1
                        else:
                            split_pos = p2

                # Split into new partitions based on split_pos
                indices_to_delete = []
                for idx, tokenized_log_entry in enumerate(p_in):
                    if tokenized_log_entry[p1] == token:
                        split_token = tokenized_log_entry[split_pos]
                        if split_token not in tmp_partitions[split_pos]:
                            tmp_partitions[split_pos][split_token] = []
                        tmp_partitions[split_pos][split_token].append(log_indices[idx])
                        indices_to_delete.append(idx)

                delete_indices_from_list(log_indices, indices_to_delete)
                delete_indices_from_list(p_in, indices_to_delete)

                if len(p_in) == 0:
                    break

            # Add new partitions in tmp_partitions to bijection_partitions
            outlier_log_indices = []
            for split_pos in tmp_partitions:
                for split_token in tmp_partitions[split_pos]:
                    child_partition = tmp_partitions[split_pos][split_token]
                    partition_support = len(child_partition) / len(partition_item.log_indices)
                    if partition_support < self.partition_threshold:
                        outlier_log_indices.extend(child_partition)
                    else:
                        bijection_partitions.add(child_partition, 3)

            if len(outlier_log_indices) > 0:
                bijection_partitions.add(outlier_log_indices, 3)

            # Split disjoint M-M groups
            while len(p_in) != 0:
                base_token = p_in[0][p1]
                mapping_finder.update_relevant_token_sets(base_token)
                domain_set = mapping_finder.domain_set
                tmp_log_indices = []
                indices_to_delete = []
                for idx, tokenized_log_entry in enumerate(p_in):
                    if tokenized_log_entry[p1] in domain_set:
                        tmp_log_indices.append(log_indices[idx])
                        indices_to_delete.append(idx)
                delete_indices_from_list(log_indices, indices_to_delete)
                delete_indices_from_list(p_in, indices_to_delete)
                bijection_partitions.add(tmp_log_indices, 3)

        self.partitions = bijection_partitions
        self._prune_partitions()

    def _get_partition_goodness(self, p_in):
        count_1 = 0
        token_count = len(p_in[0])
        for token_idx in range(token_count):
            reference_token = None
            is_unique = True
            for log_entry in p_in:
                if reference_token is None:
                    reference_token = log_entry[token_idx]
                elif reference_token != log_entry[token_idx]:
                    is_unique = False
                    break
            if is_unique:
                count_1 += 1

        return count_1 / token_count

    def _prune_partitions(self):
        """
        Place all log entries from partitions with a file support
        less than the threshold into a single partition.
        """
        # highest_pruned_step = -1
        # pruned_log_indices = []
        pruned_partitions = Partitions(self.tokenized_log_entries)
        total_line_count = len(self.tokenized_log_entries)

        for partition_item in self.partitions:
            log_indices = partition_item.log_indices
            partition_line_count = len(log_indices)
            file_support = partition_line_count / total_line_count
            # if file_support < self.file_threshold:
            #     highest_pruned_step = max(highest_pruned_step, partition_item.step)
            #     pruned_log_indices.extend(log_indices)
            # else:
            #     pruned_partitions.add(log_indices, partition_item.step)

            if file_support >= self.file_threshold:
                pruned_partitions.add(log_indices, partition_item.step)

        # if len(pruned_log_indices) > 1:
        #     pruned_partitions.add(pruned_log_indices, highest_pruned_step)

        self.partitions = pruned_partitions

    def _determine_p1_and_p2(self, tokenized_log_entries, step):
        """
        Return the indices of the two most frequent unique token count positions.
        """
        if len(tokenized_log_entries[0]) == 2:
            return 0, 1
        elif len(tokenized_log_entries[0]) > 2:
            unique_tokens = self._get_unique_tokens(tokenized_log_entries)
            if step == 2:
                card_count = {}
                for idx in unique_tokens:
                    cardinality = len(unique_tokens[idx])
                    if cardinality not in card_count:
                        card_count[cardinality] = []
                    card_count[cardinality].append(idx)
                count_per_cardinality = [(len(card_count[cardinality]), cardinality) for cardinality in card_count]
                max_count_tuples = get_n_sorted(2, count_per_cardinality, key=lambda x: x[0], get_max=True)
                if max_count_tuples[0][0] > 1:
                    max_freq_card = max_count_tuples[0][1]
                    return sorted(card_count[max_freq_card][0:2])
                elif max_count_tuples[0][0] == 1:
                    max_freq_card = max_count_tuples[0][1]
                    second_max_freq_card = max_count_tuples[1][1]
                    return sorted([card_count[max_freq_card][0], card_count[second_max_freq_card][0]])
                else:
                    raise Exception('Error trying to calculate most frequent cardinalities')
            else:
                count_per_token_idx = [(len(unique_tokens[token_idx]), token_idx) for token_idx in unique_tokens]
                counts = get_n_sorted(2, count_per_token_idx, key=lambda x: x[0], get_max=False)
                return sorted(count[1] for count in counts)
        else:
            raise Exception('Invalid log entry length')

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
        """
        Returns the split position for 1-M or M-1 scenarios.
        """
        line_count = self._get_line_count_at_token_idx(p_in, s_temp, token_idx)
        distance = len(s_temp) / line_count
        if distance < self.lower_bound:
            return 2 if is_one_to_m else 1
        else:
            return 1 if is_one_to_m else 2

    def _get_line_count_at_token_idx(self, p_in, s_temp, token_idx):
        """
        Return the number of lines with tokens from s_temp.
        """
        line_count = 0
        for log_entry in p_in:
            if log_entry[token_idx] in s_temp:
                line_count += 1
        return line_count

    def _discover_cluster_templates(self):
        """
        Discover all cluster templates from the current partitions.
        """
        for partition_item in self.partitions:
            log_indices = partition_item.log_indices
            log_entries = self._get_tokenized_log_entries_from_indices(log_indices)
            constant_token_indices = self._get_constant_token_indices(log_entries)
            cluster_template_tokens = []
            for idx in range(len(log_entries[0])):
                if idx in constant_token_indices:
                    token = log_entries[0][idx]
                    cluster_template_tokens.append(token)
                else:
                    cluster_template_tokens.append(PLACEHOLDER)
            cluster_template_string = ' '.join(cluster_template_tokens)
            self.cluster_templates[cluster_template_string] = log_indices

    def _get_constant_token_indices(self, log_entries):
        """
        Get of token indices that represent constant tokens for a given partition.
        """
        reference_tokens = log_entries[0]
        constant_token_indices = set()
        for idx in range(len(reference_tokens)):
            is_constant = True
            for log_entry in log_entries:
                if log_entry[idx] != reference_tokens[idx]:
                    is_constant = False
                    break
            if is_constant:
                constant_token_indices.add(idx)
        return constant_token_indices

    def print_cluster_templates(self):
        for template in self.cluster_templates:
            print(template)
            log_indices = self.cluster_templates[template]
            log_entries = self._get_log_entries_from_indices(log_indices)
            print_items(log_entries)
