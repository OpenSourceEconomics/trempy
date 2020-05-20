"""This module contains the class to manage the model estimation."""
import os

from trempy.shared.shared_auxiliary import criterion_function
from trempy.shared.shared_auxiliary import char_floats
from trempy.record.clsLogger import logger_obj

from trempy.config_trempy import PREFERENCE_PARAMETERS
from trempy.custom_exceptions import MaxfunError
from trempy.config_trempy import HUGE_FLOAT
from trempy.shared.clsBase import BaseCls


class EstimateClass(BaseCls):
    """This class manages all issues about the model estimation."""

    def __init__(self, df, cutoffs, questions, paras_obj, max_eval, optimizer,
                 version, **version_specific):
        """Init class."""
        self.attr = dict()
        self.attr['version'] = version

        # Handle version-specific objects outside paras_obj.
        if version in ['scaled_archimedean']:
            for key, value in version_specific.items():
                self.attr[key] = value
        elif version in ['nonstationary', 'warmglow']:
            # Currently nothing to do.
            pass

        # Initialization attributes
        self.attr['paras_obj'] = paras_obj
        self.attr['questions'] = questions
        self.attr['max_eval'] = max_eval
        self.attr['cutoffs'] = cutoffs
        self.attr['df'] = df

        # Housekeeping attributes
        self.attr['optimizer'] = optimizer
        self.attr['num_step'] = 0
        self.attr['num_eval'] = 0

        self.attr['x_econ_all_current'] = None
        self.attr['x_econ_all_start'] = None
        self.attr['x_econ_all_step'] = None

        self.attr['m_optimal_current'] = None
        self.attr['m_optimal_start'] = None
        self.attr['m_optimal_step'] = None

        self.attr['f_current'] = HUGE_FLOAT
        self.attr['f_start'] = HUGE_FLOAT
        self.attr['f_step'] = HUGE_FLOAT

        self.attr['paras_label'] = PREFERENCE_PARAMETERS[version] + questions

        self._logging_start()

    def evaluate(self, x_optim_free_current):
        """Evaluate the criterion function during an estimation."""
        # Distribute general class attributes
        paras_obj = self.attr['paras_obj']
        questions = self.attr['questions']
        version = self.attr['version']
        cutoffs = self.attr['cutoffs']
        df = self.attr['df']

        # Handle versions-specific objects outside para_obj
        if version in ['scaled_archimedean']:
            marginals = self.attr['marginals']
            upper = self.attr['upper']
            version_specific = {'upper': upper, 'marginals': marginals}
        elif version in ['nonstationary', 'warmglow']:
            version_specific = dict()

        # Construct relevant set of parameters
        paras_obj.set_values('optim', 'free', x_optim_free_current)
        x_optim_all_current = paras_obj.get_values('optim', 'all')
        x_econ_all_current = paras_obj.get_values('econ', 'all')

        # Get standard deviations. They have a larger index than nparas_econ.
        nparas_econ = paras_obj.attr['nparas_econ']
        sds = x_econ_all_current[nparas_econ:]

        fval, m_optimal = criterion_function(df, questions, cutoffs, paras_obj,
                                             version, sds, **version_specific)

        self._update_evaluation(fval, x_econ_all_current, x_optim_all_current, m_optimal)

        return fval

    def _update_evaluation(self, fval, x_econ_all_current, x_optim_all_current, m_optimal):
        """Update all attributes based on the new evaluation and write some information to file."""
        self.attr['x_econ_all_current'] = x_econ_all_current
        self.attr['m_optimal_current'] = m_optimal
        self.attr['f_current'] = fval
        self.attr['num_eval'] += 1

        # Determine special events
        is_start = self.attr['num_eval'] == 1
        is_step = fval < self.attr['f_step']

        # Record information at start
        if is_start:
            self.attr['x_econ_all_start'] = x_econ_all_current
            self.attr['m_optimal_start'] = m_optimal
            self.attr['f_start'] = fval

        # Record information at step
        if is_step:
            self.attr['x_econ_all_step'] = x_econ_all_current
            self.attr['m_optimal_step'] = m_optimal
            self.attr['f_step'] = fval
            self.attr['num_step'] += 1

        self._logging_evaluation(x_econ_all_current, x_optim_all_current)

    def _logging_start(self):
        """Record some basic properties of the estimation at the beginning."""
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

    def _logging_evaluation(self, x_econ_all_current, x_optim_all_current):
        """Manage all issues related to the logging of the estimation."""
        # Distribute attributes
        para_labels = self.attr['paras_label']
        questions = self.attr['questions']
        version = self.attr['version']

        # Update class attributes
        with open('est.trempy.info', 'w') as outfile:
            fmt_ = ' {:>10}    ' + '{:<20}    ' + '{:>25}    ' * 3

            # Write out information about criterion function
            outfile.write('\n {:<25}\n\n'.format('Criterion Function'))
            outfile.write(fmt_.format(*['', '', 'Start', 'Step', 'Current', '']) + '\n\n')
            args = (self.attr['f_start'], self.attr['f_step'], self.attr['f_current'])
            line = ['', ''] + char_floats(args) + ['']
            outfile.write(fmt_.format(*line) + '\n\n')

            # Economic Parameters
            outfile.write('\n {:<25}\n\n'.format('Economic Parameters'))
            line = ['Identifier', 'Label', 'Start', 'Step', 'Current']
            outfile.write(fmt_.format(*line) + '\n\n')
            # Handle version
            for i, _ in enumerate(range(len(questions) + len(PREFERENCE_PARAMETERS[version]))):
                line = [i]
                line += [para_labels[i]]
                # Handle optional arguments indicated by None value.
                if self.attr['x_econ_all_start'][i] is None:
                    continue
                else:
                    line += char_floats(self.attr['x_econ_all_start'][i])
                    line += char_floats(self.attr['x_econ_all_step'][i])
                    line += char_floats(self.attr['x_econ_all_current'][i])
                    outfile.write(fmt_.format(*line) + '\n')

            # Optimal Compensation
            outfile.write('\n\n {:<25}\n\n'.format('Optimal Compensations'))
            line = ['Questions', '', 'Start', 'Step', 'Current']
            outfile.write(fmt_.format(*line) + '\n\n')
            for q in questions:
                line = [q, '']
                line += char_floats(self.attr['m_optimal_start'][q])
                line += char_floats(self.attr['m_optimal_step'][q])
                line += char_floats(self.attr['m_optimal_current'][q])
                outfile.write(fmt_.format(*line) + '\n')

            # Steps and Duration
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

            fmt_ = ' {:>10}   ' + '{:<20}   ' + '{:>25}    ' * 2
            line = ['Identifier', 'Label', 'Economic', 'Optimizer']
            outfile.write(fmt_.format(*line) + '\n\n')

            for i, _ in enumerate(range(len(questions) + len(PREFERENCE_PARAMETERS[version]))):
                line = [i]
                line += [para_labels[i]]
                # Handle optional arguments with None value
                if (x_econ_all_current[i] or x_optim_all_current[i]) is None:
                    continue
                line += char_floats([x_econ_all_current[i], x_optim_all_current[i]])
                outfile.write(fmt_.format(*line) + '\n')
            # We need to keep track of captured warnings.
            logger_obj.flush(outfile)

            if version in ['scaled_archimedean']:
                # We also record the results from the fitting of the copula.
                outfile.write('\n')
                with open('fit.copulpy.info') as infile:
                    outfile.write(infile.read())
                os.remove('fit.copulpy.info')

        # We can determine the estimation if the number of requested function evaluations is
        # reached or the user requests a stop.
        is_finish = (self.attr['max_eval'] == self.attr['num_eval']) and (self.attr['max_eval'] > 1)
        is_stop = os.path.exists('.stop.trempy.scratch')
        if is_finish:
            raise MaxfunError
        if is_stop:
            os.remove('.stop.trempy.scratch')
            raise MaxfunError

    @staticmethod
    def finish(opt):
        """Collect all operations to wrap up an estimation."""
        with open('est.trempy.info', 'a') as outfile:
            outfile.write('\n {:<25}'.format('TERMINATED'))

        with open('est.trempy.log', 'a') as outfile:
            outfile.write('\n {:<25}\n'.format('OPTIMIZER RETURN'))
            outfile.write('\n Message    {:<40}'.format(str(opt['message'])))
            outfile.write('\n Success    {:<40}'.format(str(opt['success'])))
            outfile.write('\n')
