"""
Tracks details for the specified log_indices and cluster_indices.

Public functions that end in _ require the activation flag to be set to run.
"""
import numpy as np
from src.utils import get_token_counts, get_responsibilities


class Tracker:
    def __init__(self, logs, log_indices, v_indices):
        self.is_active = False
        self.log_indices = sorted(log_indices)
        self.cluster_indices = set()
        self.tokens = self._get_tokens(logs)
        self.logs = logs
        self.v_indices = v_indices
        self.log_idx_lookup = self._get_log_idx_lookup(logs, log_indices)

        self.log1 = ''
        self.log2 = ''

        self.r1_old = np.array([])
        self.r2_old = np.array([])
        self.r1_new = np.array([])
        self.r2_new = np.array([])

        self.pi_old = np.array([])
        self.theta_old = np.array([])
        self.pi_new = np.array([])
        self.theta_new = np.array([])

        self.old_target_resps = {}
        self.new_target_resps = {}

    def print_target_logs(self):
        for log_idx in self.log_indices:
            log_idx_str = '{}>'.format(log_idx)
            print('{:<5}{}'.format(log_idx_str, ' '.join(self.logs[log_idx])))
        print()

    def flag_tracking(self, link):
        self.is_active = all([self._is_log_of_interest(log) for log in link])

    def print_results_(self):
        if not self.is_active:
            return
        self.cluster_indices = self._get_dominant_clusters()
        self._print_link()
        print()
        self._print_update_step()
        print()
        self._print_responsibilies_comparison()
        print()
        self._print_parameter_comparison()

    def register_link_(self, link):
        if not self.is_active:
            return
        self.log1, self.log2 = link

    def register_old_responsibilities_(self, link):
        if not self.is_active:
            return
        r1, r2 = self._get_resp_from_link(link, self.pi_old, self.theta_old)
        self.r1_old = r1
        self.r2_old = r2

    def register_new_responsibilities_(self, link):
        if not self.is_active:
            return
        r1, r2 = self._get_resp_from_link(link, self.pi_new, self.theta_new)
        self.r1_new = r1
        self.r2_new = r2

    def register_old_parameters_(self, pi, theta):
        if not self.is_active:
            return
        self.pi_old = pi
        self.theta_old = theta

    def register_new_parameters_(self, pi, theta):
        if not self.is_active:
            return
        self.pi_new = pi
        self.theta_new = theta

    def register_old_target_responsibilities_(self):
        if not self.is_active:
            return
        self.old_target_resps = self._get_target_resps(self.pi_old,
                                                       self.theta_old)

    def register_new_target_responsibilities_(self):
        if not self.is_active:
            return
        self.new_target_resps = self._get_target_resps(self.pi_new,
                                                       self.theta_new)

    def _get_dominant_clusters(self):
        dominant_clusters = []
        for log_idx in self.log_indices:
            r = int(self.old_target_resps[log_idx].argmax())
            dominant_clusters.append(r)
        return sorted(set(dominant_clusters))

    def _is_log_of_interest(self, log):
        for token in log:
            if token not in self.tokens:
                return False
        return True

    def _get_resp_from_link(self, link, pi, theta):
        log1, log2 = link
        c1 = get_token_counts(log1, self.v_indices)
        c2 = get_token_counts(log2, self.v_indices)
        r1 = get_responsibilities(c1, pi, theta)
        r2 = get_responsibilities(c2, pi, theta)
        return r1, r2

    def _get_log_idx_lookup(self, logs, log_indices):
        log_idx_lookup = {}
        for log_idx in log_indices:
            log = ' '.join(logs[log_idx])
            log_idx_lookup[log] = log_idx
        return log_idx_lookup

    def _get_tokens(self, logs):
        tokens = set()
        for log_idx in self.log_indices:
            tokens.update(logs[log_idx])
        return tokens

    def _print_link(self):
        g1 = np.argmax(self.r1_old)
        g2 = np.argmax(self.r2_old)

        log1_str = ' '.join(self.log1)
        log2_str = ' '.join(self.log2)

        log1_idx = self.log_idx_lookup[log1_str]
        log2_idx = self.log_idx_lookup[log2_str]

        msg = 'c: {:<5}| idx: {:<5}| {}'
        print(msg.format(g1, log1_idx, log1_str))
        print('vs')
        print(msg.format(g2, log2_idx, log2_str))

    def _print_responsibilies_comparison(self):
        title_msg = '{:<5}|'.format('') + '{:^24}|' * len(self.cluster_indices)
        print(title_msg.format(*self.cluster_indices))
        for log_idx in self.old_target_resps:
            resps = []
            for cluster_idx in self.cluster_indices:
                r_old = self._zero(self.old_target_resps[log_idx][cluster_idx])
                r_new = self._zero(self.new_target_resps[log_idx][cluster_idx])
                resps.append('{:<10.4} -> {:<10.4}'.format(r_old, r_new))
            msg = '{:<5}|'.format(log_idx) + '{}|' * len(resps)
            print(msg.format(*resps))

    def _zero(self, v):
        return 0.0 if v < 0.0001 else float(v)

    def _print_parameter_comparison(self):
        msg_format_title = '{:^30}|' + '{:^28}|' * len(self.cluster_indices)
        msg_format_entries = '{:<30}|' + '{}|' * len(self.cluster_indices)
        print(msg_format_title.format('Parameter', *self.cluster_indices))

        pi_values_old = {}
        pi_values_new = {}
        for cluster_idx in self.cluster_indices:
            pi_values_old[cluster_idx] = float(self.pi_old[cluster_idx])
            pi_values_new[cluster_idx] = float(self.pi_new[cluster_idx])
        pi_comp_strs = self._get_comp_strs(pi_values_old, pi_values_new)
        msg = msg_format_entries.format('Pi', *pi_comp_strs)
        print(msg)

        for token in self.tokens:
            theta_values_old = {}
            theta_values_new = {}
            v_idx = self.v_indices[token]
            for cluster_idx in self.cluster_indices:
                theta_values_old[cluster_idx] = self.theta_old[
                    cluster_idx, v_idx]
                theta_values_new[cluster_idx] = self.theta_new[
                    cluster_idx, v_idx]
            theta_comp_strs = self._get_comp_strs(theta_values_old,
                                                  theta_values_new)
            msg = msg_format_entries.format(token, *theta_comp_strs)
            print(msg)

    def _get_target_resps(self, pi, theta):
        target_resps = {}
        for log_idx in self.log_indices:
            log = self.logs[log_idx]
            c = get_token_counts(log, self.v_indices)
            target_resps[log_idx] = get_responsibilities(c, pi, theta)
        return target_resps

    def _print_update_step(self):
        p1 = np.max(self.r1_old)
        p2 = np.max(self.r2_old)
        g1 = np.argmax(self.r1_old)
        g2 = np.argmax(self.r2_old)

        if p1 < p2:
            print('Update {}'.format(' '.join(self.log1)))
            print('From cluster {} to cluster {}'.format(g1, g2))
        else:
            print('Update {}'.format(' '.join(self.log2)))
            print('From cluster {} to cluster {}'.format(g2, g1))

    def _get_comp_strs(self, pi_values_old, pi_values_new):
        pi_value_comp_strings = []
        for cluster_idx in pi_values_old:
            v_old = pi_values_old[cluster_idx]
            v_new = pi_values_new[cluster_idx]
            pi_value_string = '{:<12.4} -> {:<12.4}'.format(v_old, v_new)
            pi_value_comp_strings.append(pi_value_string)
        return pi_value_comp_strings
