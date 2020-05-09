import contextlib
import multiprocessing as mp
from time import time
from copy import deepcopy
from global_utils import dump_results, create_file_path
from global_constants import NAME, FUNCTION


class ExperimentsPipeline:
    def __init__(self, jobs, results_dir=None):
        self.results = []
        self.n_jobs = len(jobs)
        self.results_dir = results_dir
        self.mp_jobs = self._filter_jobs(jobs, True)
        self.non_mp_jobs = self._filter_jobs(jobs, False)
        self.result_names = self._get_results_filenames(jobs, len(jobs))

    def run_experiments(self):
        with contextlib.redirect_stdout(self._get_print_output_file()):
            print('CPU count: {}\n'.format(mp.cpu_count()))
            total_time_start = time()

            jobs = self.mp_jobs + self.non_mp_jobs
            for idx, job_dict in enumerate(jobs):
                print(job_dict[NAME])
                exp_time_start = time()

                result = self._get_job_result(job_dict)

                self._print_exp_time(job_dict[NAME], exp_time_start)
                self.results.append(result)

            total_time = time() - total_time_start
            print('Total time taken: {}'.format(total_time))

    def run_experiments_mp(self):
        """
        Perform *_mp experiments separately since daemonic processes are not
        allowed to have children. Pipe all print outputs to output.txt.
        """
        with contextlib.redirect_stdout(self._get_print_output_file()):
            total_time_start = time()
            for job_dict in self.mp_jobs:
                result = self._get_job_result(job_dict)
                self.results.append(result)

            with mp.Pool(mp.cpu_count()) as pool:
                self.results.extend(pool.starmap(self._execute,
                                                 self._get_non_mp_jobs_iter()))
            total_time = time() - total_time_start
            print('Total time taken: {}'.format(total_time))

    def _get_print_output_file(self):
        path = create_file_path('output.txt', self.results_dir)
        return open(path, 'w')

    def _print_exp_time(self, name, exp_time_start):
        msg = 'Time for exp {}: {}\n'
        end_time = time() - exp_time_start
        print(msg.format(name, end_time))

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
