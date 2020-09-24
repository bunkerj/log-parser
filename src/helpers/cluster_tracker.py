import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class ClusterTracker:
    def __init__(self, logs, model, n_clusters, log_indices):
        self.model = model
        self.logs = logs
        self.log_indices_track = log_indices
        self.logs_track = self._get_tracked_logs(logs, log_indices)
        self.cluster_refs = []
        self.n_clusters = n_clusters

    def run(self, n_iter):
        self.cluster_refs = []
        for _ in range(n_iter):
            self.model.fit_single_iter()
            clusters = self.model.predict(self.logs_track)
            cluster_ref = self._get_cluster_ref(clusters)
            self.cluster_refs.append(cluster_ref)

    def show(self):
        cluster_ref_arr = np.array(self.cluster_refs)
        cluster_set = list(set(cluster_ref_arr.flatten()))
        colormap = self._get_colormap(cluster_set)
        ytick_locs = [10 * idx + 15 for idx in range(cluster_ref_arr.shape[1])]

        ax = self._construct_bars(cluster_ref_arr, colormap)
        ax.set_yticks(ytick_locs)
        ax.set_yticklabels(self.log_indices_track)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log Index')
        ax.set_title('Cluster Membership Trajectories')
        ax.legend(handles=self._get_handles(cluster_set, colormap))
        ax.grid()

        plt.show()

    def _construct_bars(self, cluster_ref_arr, colormap):
        fig, ax = plt.subplots()
        for idx in range(cluster_ref_arr.shape[1]):
            track = [(i, 1) for i in range(cluster_ref_arr.shape[0])]
            facecolors = [colormap[g] for g in cluster_ref_arr[:, idx]]
            ax.broken_barh(track, (10 * (idx + 1), 9),
                           facecolors=facecolors)
        return ax

    def _get_handles(self, cluster_set, colormap):
        handles = []
        for g in cluster_set:
            color = colormap[g]
            patch = mpatches.Patch(color=color, label=str(g))
            handles.append(patch)
        return handles

    def _get_tracked_logs(self, logs, log_indices):
        return [logs[idx] for idx in log_indices]

    def _get_cluster_ref(self, clusters):
        cluster_ref = [0] * len(self.logs_track)
        for k in clusters:
            for idx in clusters[k]:
                cluster_ref[idx] = int(k)
        return cluster_ref

    def _get_colormap(self, cluster_set):
        cm = plt.get_cmap('gist_rainbow')
        return {g: cm(1. * g / self.n_clusters) for g in cluster_set}
