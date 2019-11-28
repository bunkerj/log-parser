from src.parsers.log_parser import LogParser
from constants import PLACEHOLDER


class Node:
    def __init__(self, depth):
        self.children = {}
        self.depth = depth


class LogGroup:
    def __init__(self, first_entry, log_idx):
        self.tokenized_template = first_entry
        self.log_indices = [log_idx]

    def add(self, tokenized_log_entry, log_idx):
        """
        Adds new log entry to the group and updates the template if needed.
        """
        self.log_indices.append(log_idx)
        self._update_tokenized_template(tokenized_log_entry)

    def _update_tokenized_template(self, tokenized_log_entry):
        """
        Updates the template by adding wildcards where required.
        """
        for idx in range(len(self.tokenized_template)):
            if tokenized_log_entry[idx] != self.tokenized_template[idx]:
                self.tokenized_template[idx] = PLACEHOLDER


class Drain(LogParser):
    def __init__(self, data_config, max_depth, max_child, sim_threshold):
        super().__init__(data_config)
        self.max_depth = max_depth
        self.max_child = max_child
        self.sim_threshold = sim_threshold
        self.root = Node(0)
        self.idx = -1
        self.cluster_templates = {}
        self.log_groups = []

    def parse(self):
        """
        Inserts each log entry into the parse tree.
        """
        for idx, log_entry in enumerate(self.tokenized_log_entries):
            self.idx = idx
            self._traverse_tree(self.root)
        self._discover_cluster_templates()

    def _traverse_tree(self, node):
        """
        Recursively traverses the log tree to insert the new log entry.
        """
        log_entry = self.tokenized_log_entries[self.idx]
        if node.depth == (self.max_depth - 1) or node.depth == len(log_entry):
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
        """
        Inserts the log entry into the most similar log group.
        """
        log_entry = self.tokenized_log_entries[self.idx]
        if len(log_groups) == 0:
            log_group = LogGroup(log_entry, self.idx)
            log_groups[0] = log_group
            self.log_groups.append(log_group)
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
                log_group = LogGroup(log_entry, self.idx)
                log_groups[len(log_groups)] = log_group
                self.log_groups.append(log_group)

    def _get_similarity(self, log_entry1, log_entry2):
        """
        Returns a similarity measure between two token lists.
        """
        delta_sum = 0
        for idx in range(len(log_entry1)):
            if log_entry1[idx] == log_entry2[idx]:
                delta_sum += 1
        return delta_sum / len(log_entry1)

    def _discover_cluster_templates(self):
        """
        Discovers all cluster templates from the current partitions.
        """
        for log_group in self.log_groups:
            template = ' '.join(log_group.tokenized_template)
            self.cluster_templates[template] = log_group.log_indices

    def print_tree(self, node=None):
        """
        Print all node labels within the subtree starting at the passed node.
        """
        if node is None:
            node = self.root
        for child_key in node.children:
            if node.depth != (self.max_depth - 1):
                tab_offset = '\t' * node.depth
                print('{}{}'.format(tab_offset, child_key))
                self.print_tree(node.children[child_key])
