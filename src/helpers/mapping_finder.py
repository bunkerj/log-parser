class MappingFinder:
    def __init__(self, p1_token_mapping, p2_token_mapping):
        self.domain_set = set()
        self.codomain_set = set()
        self.p1_token_mapping = p1_token_mapping
        self.p2_token_mapping = p2_token_mapping

    def update_relevant_token_sets(self, token, parent_token=None, is_reversed=False):
        if not is_reversed:
            self.domain_set.add(token)
            for codomain_token in self.p1_token_mapping[token]:
                if parent_token is None or parent_token != codomain_token:
                    self.codomain_set.add(codomain_token)
                    self.update_relevant_token_sets(codomain_token, token, not is_reversed)
        else:
            self.codomain_set.add(token)
            for domain_token in self.p2_token_mapping[token]:
                if parent_token is None or parent_token != domain_token:
                    self.domain_set.add(domain_token)
                    self.update_relevant_token_sets(domain_token, token, not is_reversed)
