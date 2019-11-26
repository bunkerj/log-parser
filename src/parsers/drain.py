from src.parsers.log_parser import LogParser
from src.constants import PLACEHOLDER


class Node:
    def __init__(self, depth):
        self.children = {}
        self.depth = depth


class LogGroup:
    def __init__(self, first_entry, log_idx):
        self.tokenized_template = first_entry
        self.log_indices = [log_idx]

    def add(self, tokenized_log_entry, log_idx):
        self.log_indices.append(log_idx)
        self._update_tokenized_template(tokenized_log_entry)

    def _update_tokenized_template(self, tokenized_log_entry):
        for idx in range(len(self.tokenized_template)):
            if tokenized_log_entry[idx] != self.tokenized_template[idx]:
                tokenized_log_entry[idx] = PLACEHOLDER


class Drain(LogParser):
    def __init__(self, data_config, max_depth, max_child, sim_threshold):
        super().__init__(data_config)
        self.max_depth = max_depth
        self.max_child = max_child
        self.sim_threshold = sim_threshold
        self.root = Node(0)
        self.idx = -1

    def parse(self):
        for idx, log_entry in enumerate(self.tokenized_log_entries):
            self.idx = idx
            self._traverse_tree(self.root)

    def _traverse_tree(self, node):
        log_entry = self.tokenized_log_entries[self.idx]
        if node.depth == (self.max_depth - 1):
            self._update_most_similar_groups(node.children)
        elif node.depth == 0:
            child_key = len(log_entry)
            if child_key not in node.children:
                node.children[child_key] = Node(node.depth + 1)
            self._traverse_tree(node.children[child_key])
        else:
            token = log_entry[node.depth - 1]
            is_wild_card = any([
                token.isdigit(),
                PLACEHOLDER in node.children and len(node.children) == self.max_child,
                PLACEHOLDER not in node.children and len(node.children) == (self.max_child - 1),
            ])
            child_key = PLACEHOLDER if is_wild_card else token
            if child_key not in node.children:
                node.children[child_key] = Node(node.depth + 1)
            self._traverse_tree(node.children[child_key])

    def _update_most_similar_groups(self, log_groups):
        log_entry = self.tokenized_log_entries[self.idx]
        if len(log_groups) == 0:
            log_groups[0] = LogGroup(log_entry, self.idx)
        else:
            highest_sim = -1
            highest_sim_idx = -1
            for idx in log_groups:
                sim = self._get_similarity(log_entry, log_groups[idx].tokenized_template)
                if sim > highest_sim:
                    highest_sim = sim
                    highest_sim_idx = idx
            if highest_sim > self.sim_threshold:
                log_groups[highest_sim_idx].add(log_entry, self.idx)
            else:
                log_groups[len(log_groups)] = LogGroup(log_entry, self.idx)

    def _get_similarity(self, log_entry1, log_entry2):
        delta_sum = 0
        for idx in range(len(log_entry1)):
            if log_entry1[idx] == log_entry2[idx]:
                delta_sum += 1
        return delta_sum / len(log_entry1)
