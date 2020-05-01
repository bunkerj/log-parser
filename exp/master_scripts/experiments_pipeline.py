import multiprocessing as mp
from global_utils import dump_results


class ExperimentsPipeline:
    def __init__(self, jobs):
        self.results = []
        self.n_jobs = len(jobs)
        self.mp_jobs = self._filter_jobs(jobs, True)
        self.non_mp_jobs = self._filter_jobs(jobs, False)
        self.result_names = self._get_results_filenames(jobs)

    def run_experiments(self):
        """
        Perform *_mp experiments separately since daemonic processes are not
        allowed to have children.
        """
        for f in self.mp_jobs:
            args = self.mp_jobs[f]
            self.results.append(f(**args))

        with mp.Pool(mp.cpu_count()) as pool:
            self.results.extend(pool.starmap(self._execute,
                                             self.non_mp_jobs.items()))

    def write_results(self, results_dir):
        assert len(self.results) == self.n_jobs
        for idx in range(self.n_jobs):
            dump_results(self.result_names[idx],
                         self.results[idx],
                         results_dir)

    def _get_results_filenames(self, jobs):
        """
        Return array of filenames which will contain the results for each
        experiment. The filename is constructed as:
        [name of experiment function].p
        """
        return ['{}.p'.format(f.__name__) for f in jobs]

    def _filter_jobs(self, jobs, mp_flag):
        """
        Filter jobs based on whether or not we want only mp jobs (mp_flag is
        True) or only non-mp jobs (mp_flag is set to False).
        """
        keys = filter(lambda f: self._filter(f.__name__, mp_flag), jobs)
        return {k: jobs[k] for k in keys}

    def _filter(self, name, mp_flag):
        return mp_flag if name[-3:] == '_mp' else not mp_flag

    def _execute(self, f, kwargs):
        return f(**kwargs)
