"""This module contains the class to manage the model estimation."""
from trempy.shared.shared_auxiliary import criterion_function
from trempy.estimate.estimate_auxiliary import char_floats
#from interalpy.logging.clsLogger import logger_obj

from trempy.custom_exceptions import MaxfunError
from trempy.config_trempy import HUGE_FLOAT
from trempy.shared.clsBase import BaseCls


class EstimateClass(BaseCls):
    """This class manages all issues about the model estimation."""
    def __init__(self, df, cutoffs, questions, paras_obj, max_eval):

        self.attr = dict()

        # Initialization attributes
        self.attr['paras_obj'] = paras_obj
        self.attr['max_eval'] = max_eval
        self.attr['df'] = df
        self.attr['cutoffs'] = cutoffs
        self.attr['questions'] = questions

        # Housekeeping attributes
        self.attr['num_step'] = 0
        self.attr['num_eval'] = 0

        self.attr['x_econ_all_current'] = None
        self.attr['x_econ_all_start'] = None
        self.attr['x_econ_all_step'] = None

        self.attr['f_current'] = HUGE_FLOAT
        self.attr['f_start'] = HUGE_FLOAT
        self.attr['f_step'] = HUGE_FLOAT

        self.attr['paras_label'] = ['alpha', 'beta', 'eta'] + questions

        self._logging_start()

    def evaluate(self, x_optim_free_current):
        """This method allows to evaluate the criterion function during an estimation"""
        # Distribute class attributes
        paras_obj = self.attr['paras_obj']
        df = self.attr['df']
        cutoffs = self.attr['cutoffs']
        questions = self.attr['questions']

        # Construct relevant set of parameters
        paras_obj.set_values('optim', 'free', x_optim_free_current)
        x_optim_all_current = paras_obj.get_values('optim', 'all')
        x_econ_all_current = paras_obj.get_values('econ', 'all')
        fval = criterion_function(df, questions, cutoffs, *x_econ_all_current)

        self._update_evaluation(fval, x_econ_all_current, x_optim_all_current)

        return fval

    def _update_evaluation(self, fval, x_econ_all_current, x_optim_all_current):
        """This method updates all attributes based on the new evaluation and writes some
        information to files."""
        # Update current information
        self.attr['x_econ_all_current'] = x_econ_all_current
        self.attr['f_current'] = fval
        self.attr['num_eval'] += 1

        # Determine special events
        is_stop = (self.attr['max_eval'] == self.attr['num_eval']) and (self.attr['max_eval'] > 1)
        is_start = self.attr['num_eval'] == 1
        is_step = fval < self.attr['f_step']

        # Record information at start
        if is_start:
            self.attr['x_econ_all_start'] = x_econ_all_current
            self.attr['f_start'] = fval

        # Record information at step
        if is_step:
            self.attr['x_econ_all_step'] = x_econ_all_current
            self.attr['f_step'] = fval
            self.attr['num_step'] += 1

        self._logging_evaluation(is_stop, x_econ_all_current, x_optim_all_current)

    def _logging_start(self):
        """This method records some basic properties of the estimation at the beginning."""
        # Distribute class attributes
        paras_obj = self.attr['paras_obj']
        df = self.attr['df']

        # Construct auxiliary objects
        est_agents = df.index.get_level_values(0).nunique()

        with open('est.trempy.log', 'w') as outfile:
            outfile.write('\n ESTIMATION SETUP\n')

            fmt_ = '\n Agents {:>14}\n'
            outfile.write(fmt_.format(est_agents))

            outfile.write('\n PARAMETER BOUNDS\n\n')

            fmt_ = ' {:>10}   ' + '{:>25}    ' * 2
            line = ['Identifier', 'Lower', 'Upper']
            outfile.write(fmt_.format(*line) + '\n\n')
            for i, para_obj in enumerate(paras_obj.get_attr('para_objs')):

                bounds = para_obj.get_attr('bounds')
                line = [i] + char_floats(bounds)
                outfile.write(fmt_.format(*line) + '\n')

    def _logging_evaluation(self, is_stop, x_econ_all_current, x_optim_all_current):
        """This methods manages all issues related to the logging of the estimation."""
        # Distribute attributes
        para_labels = self.attr['paras_label']
        questions = self.attr['questions']

        # Update class attributes
        with open('est.trempy.info', 'w') as outfile:
            fmt_ = ' {:>10}    ' + '{:<10}    ' +'{:>25}    ' * 3

            # Write out information about criterion function
            outfile.write('\n {:<25}\n\n'.format('Criterion Function'))
            outfile.write(fmt_.format(*['', '', 'Start', 'Step', 'Current', '']) + '\n\n')
            args = (self.attr['f_start'], self.attr['f_step'], self.attr['f_current'])
            line = ['', ''] + char_floats(args) + ['']
            outfile.write(fmt_.format(*line) + '\n\n')

            outfile.write('\n {:<25}\n\n'.format('Economic Parameters'))
            line = ['Identifier', 'Label',  'Start', 'Step', 'Current']
            outfile.write(fmt_.format(*line) + '\n\n')
            for i, _ in enumerate(range(len(questions) + 3)):
                line = [i]
                line += [para_labels[i]]
                line += char_floats(self.attr['x_econ_all_start'][i])
                line += char_floats(self.attr['x_econ_all_step'][i])
                line += char_floats(self.attr['x_econ_all_current'][i])
                outfile.write(fmt_.format(*line) + '\n')

            outfile.write('\n')
            fmt_ = '\n {:<25}   {:>25}\n'
            outfile.write(fmt_.format(*['Number of Evaluations', self.attr['num_eval']]))
            outfile.write(fmt_.format(*['Number of Steps', self.attr['num_step']]))

        with open('est.trempy.log', 'a') as outfile:

            outfile.write('\n\n')
            fmt_ = '\n EVALUATION {:>10}  STEP {:>10}\n'
            outfile.write(fmt_.format(*[self.attr['num_eval'], self.attr['num_step']]))

            fmt_ = '\n Criterion {:>28}  \n\n\n'
            outfile.write(fmt_.format(char_floats(self.attr['f_current'])[0]))

            fmt_ = ' {:>10}   ' + '{:<10}   ' + '{:>25}    ' * 2
            line = ['Identifier','Label', 'Economic', 'Optimizer']
            outfile.write(fmt_.format(*line) + '\n\n')

            for i, _ in enumerate(range(len(questions) + 3)):
                line = [i]
                line += [para_labels[i]]
                line += char_floats([x_econ_all_current[i], x_optim_all_current[i]])
                outfile.write(fmt_.format(*line) + '\n')
            # We need to keep track of captured warnings.
            #logger_obj.flush(outfile)

        # We can determine the estimation if the number of requested function evaluations is
        # reached.
        if is_stop:
            raise MaxfunError

    @staticmethod
    def finish(opt):
        """This method collects all operations to wrap up an estimation."""
        with open('est.trempy.info', 'a') as outfile:
            outfile.write('\n {:<25}'.format('TERMINATED'))

        with open('est.trempy.log', 'a') as outfile:
            outfile.write('\n {:<25}\n'.format('OPTIMIZER RETURN'))
            outfile.write('\n Message    {:<25}'.format(opt['message']))
            outfile.write('\n Success    {:<25}'.format(str(opt['success'])))
            outfile.write('\n')


