"""
Version of MultinomialMixtureOnline that is enhanced to track diagnostic
information.
"""
from global_utils import get_top_k_args
from src.diagnostic.tracker import Tracker
from src.parsers.multinomial_mixture_online import MultinomialMixtureOnline


class MultinomialMixtureOnlineDiag(MultinomialMixtureOnline):
    def __init__(self, log_indices, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logs = args[0]
        self.tracker = Tracker(self.logs, log_indices, self.v_indices)

    def _enforce_must_link_constraints(self, must_links):
        print('--------------------------------------------')
        self.tracker.print_target_logs()

        for link in must_links:
            self.tracker.flag_tracking(link)
            self.tracker.register_link_(link)
            self.tracker.register_old_target_responsibilities_(
                self._get_responsibilities, self._get_token_counts)

            self._enforce_must_link_constraint(link)

            self.tracker.register_new_target_responsibilities_(
                self._get_responsibilities, self._get_token_counts)

            self.tracker.print_results_()
        print('--------------------------------------------')

    def _enforce_must_link_constraint(self, link):
        log1, log2 = link

        c1 = self._get_token_counts(log1)
        c2 = self._get_token_counts(log2)

        r1 = self._get_responsibilities(c1)
        r2 = self._get_responsibilities(c2)

        g1_first, = get_top_k_args(r1, 1)
        g2_first, = get_top_k_args(r2, 1)

        self.tracker.register_old_responsibilities_(r1, r2)
        self.tracker.register_old_parameters_(self.pi, self.theta)

        if g1_first != g2_first:
            p1_first = r1[g1_first]
            p2_first = r2[g2_first]
            if p1_first < p2_first:
                self._change_dominant_resp(c1, g1_first, g2_first)
            else:
                self._change_dominant_resp(c2, g2_first, g1_first)

        r1 = self._get_responsibilities(c1)
        r2 = self._get_responsibilities(c2)

        self.tracker.register_new_responsibilities_(r1, r2)
        self.tracker.register_new_parameters_(self.pi, self.theta)
