import functools
import numpy as np
from math import sqrt


class GreedyIterativeGeodesicAscent:
    def __init__(self, projections, M):
        self.projections = self._filter_projs(projections)
        self.N = len(projections)
        self.M = M

    def get_weights(self):
        L = self._get_reference_vector()
        l = L / self._norm(L)
        norm_projs = self._get_norm_projections()
        w = np.zeros((self.N, 1))
        l_w = np.zeros(norm_projs[0].shape)

        for _ in range(self.M):
            d_ref, dirs = self._get_geodesic_dirs(l, l_w, norm_projs)
            n = self._get_best_geodesic(d_ref, dirs)
            step_size = self._get_step_size(l, l_w, norm_projs, n)
            l_w, w = self._update_coreset(l_w, w, norm_projs, n, step_size)

        return self._scale_weights(w, l_w, l, L)

    def _get_reference_vector(self):
        return functools.reduce(lambda a, b: a + b, self.projections)

    def _get_norm_projections(self):
        return list(map(lambda v: v / self._norm(v), self.projections))

    def _norm(self, v):
        return sqrt(v.T @ v)

    def _get_geodesic_dir(self, l_candidate, l_w):
        numerator = l_candidate - (l_candidate.T @ l_w) * l_w
        denominator = self._norm(numerator)
        return numerator / denominator if denominator > 0 \
            else np.zeros(l_w.shape)

    def _get_candidate_directions(self, l_w, norm_projs):
        return [self._get_geodesic_dir(l_n, l_w) for l_n in norm_projs]

    def _get_geodesic_dirs(self, l, l_w, norm_projections):
        d_ref = self._get_geodesic_dir(l, l_w)
        dirs = self._get_candidate_directions(l_w, norm_projections)
        return d_ref, dirs

    def _get_best_geodesic(self, d_ref, dirs):
        similarities = np.array([d_ref.T @ d_n for d_n in dirs])
        return similarities.argmax()

    def _get_step_size(self, l, l_w, norm_projections, n):
        l_n = norm_projections[n]
        c_0 = l.T @ l_n
        c_1 = l.T @ l_w
        c_2 = l_n.T @ l_w
        return (c_0 - c_1 * c_2) / ((c_0 - c_1 * c_2) + (c_1 - c_0 * c_2))

    def _update_coreset(self, l_w, w, norm_projs, n, step_size):
        l_n = norm_projs[n]

        l_w_num = (1 - step_size) * l_w + step_size * l_n
        l_w_denom = self._norm(l_w_num)
        l_w = l_w_num / l_w_denom

        one_n = self._get_basis(n)
        w_num = (1 - step_size) * w + step_size * one_n
        w = w_num / l_w_denom

        return l_w, w

    def _get_basis(self, n):
        one_n = np.zeros((self.N, 1))
        one_n[n] = 1
        return one_n

    def _scale_weights(self, w, l_w, l, L):
        L_norm = self._norm(L)
        for n in range(self.N):
            L_n = self.projections[n]
            L_n_norm = self._norm(L_n)
            w[n] *= float((L_norm / L_n_norm) * (l_w.T @ l))
        return w

    def _filter_projs(self, projs):
        return list(filter(lambda v_n: abs(self._norm(v_n)) > 0, projs))
