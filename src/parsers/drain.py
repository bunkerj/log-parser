from copy import copy
from src.utils import has_digit
from global_constants import PLACEHOLDER
from src.parsers.base.log_parser import LogParser


class Node:
    def __init__(self, depth):
        self.children = {}
        self.depth = depth


class LogGroup:
    def __init__(self):
        self.tokenized_template = None
        self.log_indices = []

    def add(self, tokenized_log, log_idx):
        """
        Adds new log to the group and updates the template if needed.
        """
        self.log_indices.append(log_idx)
        self._update_tokenized_template(tokenized_log)

    def _update_tokenized_template(self, tokenized_log):
        """
        Updates the template by adding wildcards where required.
        """
        if self.tokenized_template is None:
            self.tokenized_template = copy(tokenized_log)
        else:
            for idx in range(len(self.tokenized_template)):
                if tokenized_log[idx] != self.tokenized_template[idx]:
                    self.tokenized_template[idx] = PLACEHOLDER


class Drain(LogParser):
    def __init__(self, tokenized_logs, max_depth, max_child,
                 sim_threshold):
        super().__init__(tokenized_logs)
        self.max_depth = round(max_depth)
        self.max_child = round(max_child)
        self.sim_threshold = sim_threshold
        self.root = Node(0)
        self.idx = -1
        self.cluster_templates = {}
        self.log_groups = []

    def parse(self):
        """
        Inserts each log into the parse tree.
        """
        for idx in range(len(self.tokenized_logs)):
            self.single_parse()
        self.discover_cluster_templates()

    def single_parse(self):
        self.idx += 1
        self._traverse_tree(self.root)

    def discover_cluster_templates(self):
        """
        Discovers all cluster templates from the current partitions.
        """
        self.cluster_templates = {}
        for log_group in self.log_groups:
            template = ' '.join(log_group.tokenized_template)
            self.cluster_templates[template] = log_group.log_indices

    def print_tree(self, node=None):
        """
        Print all node labels within the subtree starting at the passed node.
        """
        if node is None:
            node = self.root
        if hasattr(node, 'children'):
            for child_key in node.children:
                child_node = node.children[child_key]
                if child_node.__class__.__name__ == 'Node':
                    tab_offset = '\t' * node.depth
                    print('{}{}'.format(tab_offset, child_key))
                    self.print_tree(child_node)

    def _traverse_tree(self, node):
        """
        Recursively traverses the log tree to insert the new log.
        """
        log = self.tokenized_logs[self.idx]
        if node.depth == (self.max_depth - 1) or node.depth == (
                len(log) + 1):
            self._update_most_similar_groups(node.children)
        elif node.depth == 0:
            child_key = len(log)
            if child_key not in node.children:
                node.children[child_key] = Node(node.depth + 1)
            self._traverse_tree(node.children[child_key])
        else:
            token = log[node.depth - 1]
            is_wild_card = any([
                has_digit(token),
                PLACEHOLDER in node.children and len(
                    node.children) == self.max_child,
                PLACEHOLDER not in node.children and len(node.children) == (
                        self.max_child - 1),
            ])
            child_key = PLACEHOLDER if is_wild_card else token
            if child_key not in node.children:
                node.children[child_key] = Node(node.depth + 1)
            self._traverse_tree(node.children[child_key])

    def _update_most_similar_groups(self, log_groups):
        """
        Inserts the log into the most similar log group.
        """
        log = self.tokenized_logs[self.idx]
        if len(log_groups) == 0:
            log_group = LogGroup()
            log_group.add(log, self.idx)
            log_groups[0] = log_group
            self.log_groups.append(log_group)
        else:
            highest_sim = -1
            highest_sim_idx = -1
            for idx in log_groups:
                sim = self._get_similarity(log_groups[idx].tokenized_template,
                                           log)
                if sim > highest_sim:
                    highest_sim = sim
                    highest_sim_idx = idx
            if highest_sim >= self.sim_threshold:
                log_groups[highest_sim_idx].add(log, self.idx)
            else:
                log_group = LogGroup()
                log_group.add(log, self.idx)
                log_groups[len(log_groups)] = log_group
                self.log_groups.append(log_group)

    def _get_similarity(self, template, log):
        """
        Returns a similarity measure between two token lists.
        """
        delta_sum = 0
        for idx in range(len(template)):
            if template[idx] != PLACEHOLDER and template[idx] == log[idx]:
                delta_sum += 1
        return delta_sum / len(template)
