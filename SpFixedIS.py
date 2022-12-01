import logging
import numpy as np
import time
#######################################
#######################################


class SpFixedISLog:
    def __init__(self, is_debug):
        # if is_debug is set to True, debug information will also be printed
        self.is_debug = is_debug
        # create logger
        self.logger = logging.getLogger('SpFixedIS')
        # create console handler
        self.ch = logging.StreamHandler()
        # clear the logger to avoid duplicate logs
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        ####
        # set level
        if self.is_debug:
            self.logger.setLevel(logging.DEBUG)
            self.ch.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
            self.ch.setLevel(logging.INFO)
        ####
        # create formatter
        self.formatter = logging.Formatter(fmt='{name}-{levelname}: {message}', style='{')
        # add formatter to console handler
        self.ch.setFormatter(self.formatter)
        # add console handler to logger
        self.logger.addHandler(self.ch)

#######################################
#######################################


class SpFixedISKernel:

    def __init__(self, params):
        """
        algorithm parameters initialization
        """
        self._perturb_amount = 0.05
        ####
        self._gain_min = 0.01
        self._gain_max = 1.0
        ####
        self._change_min = 0.0
        self._change_max = 0.5
        ####
        self._bb_bottom_threshold = 1e-5
        ####
        self._mon_gain_A = 100
        self._mon_gain_a = 0.75
        self._mon_gain_alpha = 0.6
        #####
        self._gain_type = params['gain_type']
        self._num_instances_selected = params['num_instances']
        self._iter_max = params['iter_max']
        self._stall_limit = params['stall_limit']
        self._logger = SpFixedISLog(params['is_debug'])
        self._same_count_max = self._iter_max
        self._stall_tolerance = params.get('stall_tolerance')
        self._instances_to_keep_indices = params['instances_to_keep_indices']
        self._num_gain_smoothing = params['num_gain_smoothing']
        self._print_freq = params.get('print_freq')
        self._random_state = params.get('random_state')
        self._decimals = params.get('display_rounding')
        ####
        self._num_grad_avg = params['num_grad_avg']
        ####
        self._input_x = None
        self._output_y = None
        self._wrapper = None
        self._scoring = None
        ####
        self._curr_imp_prev = None
        self._imp = None
        self._ghat = None
        self._cv_feat_eval = None
        self._cv_grad_avg = None
        self._curr_imp = None
        ####
        self._p = None
        self._n = None
        self._stall_counter = 1
        self._run_time = -1
        self._best_iter = -1
        self._gain = -1
        self._best_value = -1 * np.inf
        self._selected_instances = list()
        self._selected_instances_prev = list()
        self._best_instances = list()
        self._best_imps = list()
        self._best_imps_full = list()
        self._raw_gain_seq = list()
        self._iter_results = self.prepare_results_dict()

    def set_inputs(self, x, y, wrapper, scoring):
        self._input_x = x
        self._output_y = y
        self._wrapper = wrapper
        self._scoring = scoring

    @staticmethod
    def prepare_results_dict():
        iter_results = dict()
        iter_results['values'] = list()
        iter_results['gains'] = list()
        iter_results['gains_raw'] = list()
        iter_results['importances'] = list()
        iter_results['instance_indices'] = list()

        iter_results['iter_accuracy'] = list()
        iter_results['iter_instances'] = list()
        return iter_results

    def init_parameters(self, initial_imp=np.array([])):
        self._p = self._input_x.shape[1]
        self._n = self._input_x.shape[0]

        if len(initial_imp) == 0:
            self._curr_imp = np.repeat(0.0, self._n)
        else:
            self._curr_imp = initial_imp

        self._ghat = np.repeat(0.0, self._n)
        self._curr_imp_prev = self._curr_imp

    def print_algo_info(self):
        self._logger.logger.info(f'Wrapper: {self._wrapper}')
        self._logger.logger.info(f'Scoring metric: {str(self._scoring).split()[1]}')
        self._logger.logger.info(f"Number of features: {self._p}")
        self._logger.logger.info(f"Number of instances: {self._n}")
        self._logger.logger.info(f"Number of instances to selected: {self._num_instances_selected}")

    def get_selected_instances(self, imp):
        """
        given the importance array, determine which instances to select (as indices)
        :param imp: importance array
        :return: indices of selected instances
        """
        selected_instances = imp.copy()  # init_parameters
        
        if self._instances_to_keep_indices is not None:
            selected_instances[self._instances_to_keep_indices] = 1.0  # keep these for sure by setting their imp to 1
    
        if self._num_instances_selected < 1:
            raise ValueError('Number of instances to select must be positive.')

        else:  # user-supplied _num_instances_selected
            if self._instances_to_keep_indices is None:
                num_instances_to_keep = 0
            else:
                num_instances_to_keep = len(self._instances_to_keep_indices)
    
            num_instances_to_select = \
                np.minimum(self._n, (num_instances_to_keep + self._num_instances_selected))
    
        return selected_instances.argsort()[::-1][:num_instances_to_select]

    def eval_instance_set(self, c_imp):
        """
        given the importance array, evaluate the selected instances using the non-selected instances
        :param c_imp: importance array
        :return: performance of selected instances
        """
        selected_instances = self.get_selected_instances(c_imp)
        x_is = self._input_x[selected_instances, :]
        y_is = self._output_y[selected_instances]

        # if only 1 class is present, reinitialise the search
        if len(np.unique(y_is)) == 1:
            self.init_parameters()
            return 0

        # train with the selected instances
        self._wrapper.fit(x_is, y_is)

        # predict rest of the data and compute performance
        unselected_instances = np.arange(self._n)[np.setdiff1d(np.arange(self._n), selected_instances)]
        x_test = self._input_x[unselected_instances, :]
        y_test = self._output_y[unselected_instances]
        # x_test = self._input_x
        # y_test = self._output_y

        y_predict = self._wrapper.predict(x_test)

        best_value_mean = np.round(self._scoring(y_test, y_predict), 4)

        return best_value_mean

    def clip_change(self, raw_change):
        change_sign = np.where(raw_change > 0.0, +1, -1)
        change_abs_clipped = np.abs(raw_change).clip(min=self._change_min, max=self._change_max)
        change_clipped = change_sign * change_abs_clipped
        return change_clipped

    def run_kernel(self):
        np.random.seed(self._random_state)
        start_time = time.time()
        curr_iter_no = -1
        while curr_iter_no < self._iter_max:
            curr_iter_no += 1

            g_matrix = np.array([]).reshape(0, self._n)

            curr_imp_sel_ft_sorted = np.sort(self.get_selected_instances(self._curr_imp))

            # gradient averaging
            # make sure y_plus and y_plus are different so that this is a proper gradient
            valid_grad_counter = 0
            bad_grad_counter = 0

            for grad_iter in range(self._num_grad_avg):

                # make sure plus/ minus perturbation vectors are different from the current vector
                bad_perturb_counter = 0
                while bad_perturb_counter < self._stall_limit:
                    bad_perturb_counter += 1

                    delta = np.where(np.random.sample(self._n) >= 0.5, 1, -1)

                    imp_plus = self._curr_imp + self._perturb_amount * delta
                    imp_minus = self._curr_imp - self._perturb_amount * delta

                    imp_plus_sel_ft_sorted = np.sort(self.get_selected_instances(imp_plus))
                    imp_minus_sel_ft_sorted = np.sort(self.get_selected_instances(imp_minus))

                    if not (np.array_equal(curr_imp_sel_ft_sorted, imp_plus_sel_ft_sorted) and
                            np.array_equal(curr_imp_sel_ft_sorted, imp_minus_sel_ft_sorted)):
                        break
                    else:
                        bad_grad_counter += 1

                if bad_perturb_counter > 0:
                    self._logger.logger.debug(f'=> iter_no: {curr_iter_no}, bad_perturb_counter: '
                                              f'{bad_perturb_counter} at gradient iteration {grad_iter}')

                y_plus = self.eval_instance_set(imp_plus)
                y_minus = self.eval_instance_set(imp_minus)

                if y_plus != y_minus:
                    valid_grad_counter += 1
                    g_curr = (y_plus - y_minus) / (2 * self._perturb_amount * delta)
                    g_matrix = np.vstack([g_matrix, g_curr])
                else:
                    self._logger.logger.debug(f'=> iter_no: {curr_iter_no}, '
                                              f'y_plus == y_minus at gradient iteration {grad_iter}')
                    if bad_grad_counter >= int(self._stall_limit / 3):
                        # we give up after 1/3 of the stall limit, and we will initialize the search
                        break

            if g_matrix.shape[0] < self._num_grad_avg:
                self._logger.logger.debug(f'=> iter_no: {curr_iter_no}, '
                                          f'zero gradient(s) encountered: only {g_matrix.shape[0]} gradients averaged.')

            ghat_prev = self._ghat.copy()

            if len(g_matrix) == 0:
                self._logger.logger.debug(f'=> iter_no: {curr_iter_no}, '
                                          f'no proper gradient found, searching in the previous direction.')
                self._ghat = ghat_prev
            else:
                g_matrix_avg = g_matrix.mean(axis=0)
                if np.count_nonzero(g_matrix_avg) == 0:
                    self._logger.logger.debug(f'=> iter_no: {curr_iter_no}, '
                                              f'zero gradient encountered, searching in the previous direction.')
                    self._ghat = ghat_prev
                else:
                    self._ghat = g_matrix_avg

            if self._gain_type == 'bb':
                if curr_iter_no == 0:
                    self._gain = self._gain_min
                    self._raw_gain_seq.append(self._gain)
                else:
                    imp_diff = self._curr_imp - self._curr_imp_prev
                    ghat_diff = self._ghat - ghat_prev
                    bb_bottom = -1 * np.sum(imp_diff * ghat_diff)  # -1 due to maximization in SPSA
                    # make sure we don't end up with division by zero
                    # or negative gains:
                    if bb_bottom < self._bb_bottom_threshold:
                        self._gain = self._gain_min
                    else:
                        self._gain = np.sum(imp_diff * imp_diff) / bb_bottom
                        self._gain = np.maximum(self._gain_min, (np.minimum(self._gain_max, self._gain)))
                    self._raw_gain_seq.append(self._gain)
                    if curr_iter_no >= self._num_gain_smoothing:
                        raw_gain_seq_recent = self._raw_gain_seq[-self._num_gain_smoothing:]
                        self._gain = np.mean(raw_gain_seq_recent)
            elif self._gain_type == 'mon':
                self._gain = self._mon_gain_a / ((curr_iter_no + self._mon_gain_A) ** self._mon_gain_alpha)
                self._raw_gain_seq.append(self._gain)
            else:
                raise ValueError('Error: unknown gain type')

            self._logger.logger.debug(f'iteration gain raw = {self._raw_gain_seq[-1]:1.4f}')
            self._logger.logger.debug(f'iteration gain smooth = {self._gain:1.4f}')

            self._curr_imp_prev = self._curr_imp.copy()

            # make sure change is not too much
            curr_change_raw = self._gain * self._ghat
            self._logger.logger.debug(f"curr_change_raw = {np.round(curr_change_raw, self._decimals)}")
            curr_change_clipped = self.clip_change(curr_change_raw)
            self._logger.logger.debug(f"curr_change_clipped = {np.round(curr_change_clipped, self._decimals)}")

            # we use "+" below so that SPSA maximizes
            self._curr_imp = self._curr_imp + curr_change_clipped

            self._selected_instances_prev = self.get_selected_instances(self._curr_imp_prev)
            self._selected_instances = self.get_selected_instances(self._curr_imp)

            sel_ft_prev_sorted = np.sort(self._selected_instances_prev)

            same_instance_counter = 0
            curr_imp_orig = self._curr_imp.copy()
            same_instance_step_size = (self._gain_max - self._gain_min) / self._stall_limit
            while np.array_equal(sel_ft_prev_sorted, np.sort(self._selected_instances)):
                same_instance_counter = same_instance_counter + 1
                curr_step_size = (self._gain_min + same_instance_counter * same_instance_step_size)
                curr_change_raw = curr_step_size * self._ghat
                curr_change_clipped = self.clip_change(curr_change_raw)
                self._curr_imp = curr_imp_orig + curr_change_clipped
                self._selected_instances = self.get_selected_instances(self._curr_imp)
                if same_instance_counter >= self._stall_limit:
                    break

            if same_instance_counter > 1:
                self._logger.logger.debug(f"same_instance_counter = {same_instance_counter}")

            fs_perf_output = self.eval_instance_set(self._curr_imp)

            self._iter_results['values'].append(round(fs_perf_output, self._decimals))
            self._iter_results['gains'].append(round(self._gain, self._decimals))
            self._iter_results['gains_raw'].append(round(self._raw_gain_seq[-1], self._decimals))
            self._iter_results['importances'].append(self._curr_imp)
            self._iter_results['instance_indices'].append(self._selected_instances)
            self._iter_results['iter_instances'].append(len(self._selected_instances))

            if self._iter_results['values'][curr_iter_no] >= self._best_value + self._stall_tolerance:
                self._stall_counter = 1
                self._best_iter = curr_iter_no
                self._best_value = self._iter_results['values'][curr_iter_no]
                self._best_instances = self._selected_instances
                self._best_imps = self._curr_imp[self._best_instances]
                self._best_imps_full = self._curr_imp
            else:
                self._stall_counter = self._stall_counter + 1

            if curr_iter_no % self._print_freq == 0:
                self._logger.logger.info(f"iter: {curr_iter_no}, value: {self._iter_results['values'][curr_iter_no]}, "
                                         f"num. instances: {len(self._selected_instances)}, "
                                         f"best value: {self._best_value}")

            if bad_grad_counter >= self._stall_limit:
                # search stalled, start from scratch!
                self._logger.logger.info(f"bad gradient counter limit reached, initializing search...")
                self._stall_counter = 1  # reset the stall counter
                self.init_parameters()  # set _curr_imp and _g_hat to vectors of zeros

            if self._stall_counter > self._stall_limit:
                # search stalled, start from scratch!
                self._logger.logger.info(f"iteration stall limit reached, initializing search...")
                self._stall_counter = 1  # reset the stall counter
                self.init_parameters()  # set _curr_imp and _g_hat to vectors of zeros

            if same_instance_counter >= self._stall_limit:
                # search stalled, start from scratch!
                self._logger.logger.info(f"same instance counter limit reached, initializing search...")
                self._stall_counter = 1  # reset the stall counter
                self.init_parameters()
        self._run_time = round((time.time() - start_time) / 60, 2)  # report time in minutes
        self._logger.logger.info(f"SpFixedIS completed in {self._run_time} minutes.")
        self._logger.logger.info(f"SpFixedIS run completed.")

    def parse_results(self):
        selected_data = self._input_x[self._best_instances, :]
        results_values = np.array(self._iter_results.get('values'))
        total_iter_for_opt = np.argmax(results_values)

        return {'wrapper': self._wrapper,
                'scoring': self._scoring,
                'selected_data': selected_data,
                'iter_results': self._iter_results,
                'instances': self._best_instances,
                'importance': self._best_imps,
                'importance_full': self._best_imps_full,
                'num_instances': len(self._best_instances),
                'total_iter_overall': len(self._iter_results.get('values')),
                'total_iter_for_opt': total_iter_for_opt,
                'best_value': self._best_value,
                'current_importance': self._curr_imp,
                'current_value':  self._iter_results['values'][self._iter_max],
                'iter_accuracy': self._iter_results['iter_accuracy'],
                'iter_instances': self._iter_results['iter_instances'],
                'run_time': self._run_time
                }

#######################################
#######################################


class SpFixedIS:
    def __init__(self, x, y, wrapper, scoring):
        self._x = x
        self._y = y
        self._wrapper = wrapper
        self._scoring = scoring
        self.results = None

    def run(self,
            num_instances,
            iter_max=100,
            stall_limit=35,
            num_grad_avg=10,
            num_gain_smoothing=1,
            stall_tolerance=1e-8,
            print_freq=10,
            display_rounding=5,
            random_state=999,
            instances_to_keep_indices=None,
            is_debug=False):

        # define a dictionary to initialize the SpFixedIS kernel
        sp_params = dict()

        sp_params['num_instances'] = num_instances
        sp_params['iter_max'] = iter_max
        sp_params['stall_limit'] = stall_limit
        sp_params['num_grad_avg'] = num_grad_avg
        sp_params['num_gain_smoothing'] = num_gain_smoothing
        sp_params['stall_tolerance'] = stall_tolerance
        sp_params['print_freq'] = print_freq
        sp_params['random_state'] = random_state
        sp_params['instances_to_keep_indices'] = instances_to_keep_indices
        sp_params['is_debug'] = is_debug
        sp_params['display_rounding'] = display_rounding

        # *** for advanced users ***
        # two gain types are available: bb (barzilai & borwein) or mon (monotone)
        sp_params['gain_type'] = 'bb'

        kernel = SpFixedISKernel(sp_params)

        kernel.set_inputs(x=self._x,
                          y=self._y,
                          wrapper=self._wrapper,
                          scoring=self._scoring)

        kernel.init_parameters()
        kernel.print_algo_info()
        kernel.run_kernel()

        self.results = kernel.parse_results()

        return self
