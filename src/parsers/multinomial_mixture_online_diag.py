"""
Version of MultinomialMixtureOnline that is enhanced to track diagnostic
information.
"""
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

            self.tracker.register_old_parameters_(self.pi, self.theta)
            self.tracker.register_link_(link)
            self.tracker.register_old_target_responsibilities_()
            self.tracker.register_old_responsibilities_(link)

            self._enforce_must_link_constraint(link)

            self.tracker.register_new_parameters_(self.pi, self.theta)
            self.tracker.register_new_responsibilities_(link)
            self.tracker.register_new_target_responsibilities_()

            self.tracker.print_results_()
        print('--------------------------------------------')
