import multiprocessing as mp
from copy import deepcopy
from global_utils import dump_results
from global_constants import NAME, FUNCTION


class ExperimentsPipeline:
    def __init__(self, jobs):
        self.results = []
        self.n_jobs = len(jobs)
        self.mp_jobs = self._filter_jobs(jobs, True)
        self.non_mp_jobs = self._filter_jobs(jobs, False)
        self.result_names = self._get_results_filenames(jobs, len(jobs))

    def run_experiments(self):
        jobs = self.mp_jobs + self.non_mp_jobs
        for job_dict in jobs:
            result = self._get_job_result(job_dict)
            self.results.append(result)

    def run_experiments_mp(self):
        """
        Perform *_mp experiments separately since daemonic processes are not
        allowed to have children.
        """
        for job_dict in self.mp_jobs:
            result = self._get_job_result(job_dict)
            self.results.append(result)

        with mp.Pool(mp.cpu_count()) as pool:
            self.results.extend(pool.starmap(self._execute,
                                             self._get_non_mp_jobs_iter()))

    def write_results(self, results_dir):
        for idx in range(self.n_jobs):
            dump_results(self.result_names[idx],
                         self.results[idx],
                         results_dir)

    def _get_job_result(self, job_dict):
        f = job_dict[FUNCTION]
        kwargs = self._get_kwargs(job_dict)
        return f(**kwargs)

    def _get_results_filenames(self, jobs, n_jobs):
        """
        Return array of filenames which will contain the results for each
        experiment. The filename is constructed as:
        [name of experiment function].p
        """
        result_names = ['{}.p'.format(job_dict[NAME]) for job_dict in jobs]
        assert len(set(result_names)) == n_jobs
        return result_names

    def _filter_jobs(self, jobs, mp_flag):
        """
        Filter jobs based on whether or not we want only mp jobs (mp_flag is
        True) or only non-mp jobs (mp_flag is set to False).
        """
        return list(filter(lambda job: self._filter(job, mp_flag), jobs))

    def _filter(self, job, mp_flag):
        return mp_flag if job[NAME][-3:] == '_mp' else not mp_flag

    def _execute(self, job_dict):
        f = job_dict[FUNCTION]
        kwargs = self._get_kwargs(job_dict)
        return f(**kwargs)

    def _get_kwargs(self, job_dict):
        exp_dict_copy = deepcopy(job_dict)
        exp_dict_copy.pop(NAME, None)
        exp_dict_copy.pop(FUNCTION, None)
        return exp_dict_copy

    def _get_non_mp_jobs_iter(self):
        return [(job_dict,) for job_dict in self.non_mp_jobs]
