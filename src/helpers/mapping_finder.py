from copy import deepcopy


class MappingFinder:
    def __init__(self, p1_token_mapping, p2_token_mapping):
        self.domain_set = set()
        self.codomain_set = set()
        self.p1_token_mapping = deepcopy(p1_token_mapping)
        self.p2_token_mapping = deepcopy(p2_token_mapping)

    def update_relevant_token_sets(self, token, is_reversed=False):
        self.domain_set = set()
        self.codomain_set = set()
        self._update_relevant_token_sets(token, is_reversed)

    def _update_relevant_token_sets(self, token, is_reversed):
        if not is_reversed:
            self.domain_set.add(token)
            for codomain_token in self.p1_token_mapping[token]:
                if codomain_token not in self.codomain_set:
                    self.codomain_set.add(codomain_token)
                    self._update_relevant_token_sets(codomain_token,
                                                     not is_reversed)
        else:
            self.codomain_set.add(token)
            for domain_token in self.p2_token_mapping[token]:
                if domain_token not in self.domain_set:
                    self.domain_set.add(domain_token)
                    self._update_relevant_token_sets(domain_token,
                                                     not is_reversed)
