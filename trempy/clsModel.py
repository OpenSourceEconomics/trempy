"""This module includes the specification of the model."""
import numpy as np

from trempy.custom_exceptions import TrempyError
from trempy.shared.shared_auxiliary import dist_class_attributes
from trempy.shared.shared_auxiliary import print_init_dict
from trempy.config_trempy import PREFERENCE_PARAMETERS
from trempy.paras.clsParas import ParasCls
from trempy.shared.clsBase import BaseCls
from trempy.read.read import read


class ModelCls(BaseCls):
    """This class manages all issues about the model specification."""

    def __init__(self, fname):
        """Init class."""
        init_dict = read(fname)
        version = init_dict['VERSION']['version']

        # We first tackle the more complex issue of parameter management.
        self.attr = dict()
        self.attr['version'] = version

        # Parameters
        paras_obj = ParasCls(init_dict)
        self.attr['paras_obj'] = paras_obj

        # Version specific parameters that don't change during estimation.
        if version in ['scaled_archimedean']:
            # Information
            upper = []
            upper += [init_dict['UNIATTRIBUTE SELF']['max']]
            upper += [init_dict['UNIATTRIBUTE OTHER']['max']]
            self.attr['upper'] = upper

            # Marginal utility functions
            marginals = []
            marginals += [init_dict['UNIATTRIBUTE SELF']['marginal']]
            marginals += [init_dict['UNIATTRIBUTE OTHER']['marginal']]
            self.attr['marginals'] = marginals
        elif version in ['nonstationary']:
            pass
        else:
            raise TrempyError('version not implemented')

        # Cutoffs
        self.attr['cutoffs'] = init_dict['CUTOFFS']

        # Simulation
        self.attr['sim_agents'] = init_dict['SIMULATION']['agents']
        self.attr['sim_seed'] = init_dict['SIMULATION']['seed']
        self.attr['sim_file'] = init_dict['SIMULATION']['file']

        # Estimation
        self.attr['est_detailed'] = init_dict['ESTIMATION']['detailed']
        self.attr['optimizer'] = init_dict['ESTIMATION']['optimizer']

        self.attr['est_agents'] = init_dict['ESTIMATION']['agents']
        self.attr['num_skip'] = init_dict['ESTIMATION']['skip']
        self.attr['est_file'] = init_dict['ESTIMATION']['file']
        self.attr['maxfun'] = init_dict['ESTIMATION']['maxfun']
        self.attr['start'] = init_dict['ESTIMATION']['start']

        # Optimizer options
        self.attr['opt_options'] = dict()

        self.attr['opt_options']['SCIPY-BFGS'] = dict()
        self.attr['opt_options']['SCIPY-BFGS']['gtol'] = init_dict['SCIPY-BFGS']['gtol']
        self.attr['opt_options']['SCIPY-BFGS']['eps'] = init_dict['SCIPY-BFGS']['eps']

        self.attr['opt_options']['SCIPY-POWELL'] = dict()
        self.attr['opt_options']['SCIPY-POWELL']['xtol'] = init_dict['SCIPY-POWELL']['xtol']
        self.attr['opt_options']['SCIPY-POWELL']['ftol'] = init_dict['SCIPY-POWELL']['ftol']

        para_objs = paras_obj.get_attr('para_objs')

        questions = []
        for para_obj in para_objs:
            label = para_obj.get_attr('label')
            if label in PREFERENCE_PARAMETERS[version]:
                continue
            else:
                questions += [label]

        self.attr['questions'] = sorted(questions)
        self.attr['num_questions'] = len(questions)

        # We now need to check the integrity of the class instance.
        self._check_integrity()

    def update(self, perspective, which, values):
        """Update the estimation parameters."""
        # Distribute class attributes
        paras_obj = self.attr['paras_obj']

        paras_obj.set_values(perspective, which, values)

    def write_out(self, fname):
        """Create a initialization dictionary of the current class instance."""
        init_dict = dict()

        version = self.get_attr('version')
        paras_obj = self.attr['paras_obj']
        questions = self.attr['questions']

        # Group block labels: basis labels and version specific labels.
        basis_labels = ['VERSION', 'SIMULATION', 'ESTIMATION', 'SCIPY-BFGS',
                        'SCIPY-POWELL', 'CUTOFFS', 'QUESTIONS']
        version_labels = []
        if version in ['scaled_archimedean']:
            version_labels += ['UNIATTRIBUTE SELF', 'UNIATTRIBUTE OTHER', 'MULTIATTRIBUTE COPULA']
        elif version in ['nonstationary']:
            version_labels += ['ATEMPORAL', 'DISCOUNTING']

        # Create init dictionary
        for label in basis_labels + version_labels:
            init_dict[label] = dict()

        # Fill dictionary

        # 1) Version
        init_dict['VERSION']['version'] = version

        # 2) Simulation
        init_dict['SIMULATION']['agents'] = self.attr['sim_agents']
        init_dict['SIMULATION']['seed'] = self.attr['sim_seed']
        init_dict['SIMULATION']['file'] = self.attr['sim_file']

        # 3) Estimation
        init_dict['ESTIMATION']['detailed'] = self.attr['est_detailed']
        init_dict['ESTIMATION']['optimizer'] = self.attr['optimizer']
        init_dict['ESTIMATION']['agents'] = self.attr['est_agents']
        init_dict['ESTIMATION']['skip'] = self.attr['num_skip']
        init_dict['ESTIMATION']['file'] = self.attr['est_file']
        init_dict['ESTIMATION']['maxfun'] = self.attr['maxfun']
        init_dict['ESTIMATION']['start'] = self.attr['start']

        # 4+5) Optimizer options
        init_dict['SCIPY-BFGS'] = dict()
        init_dict['SCIPY-BFGS']['gtol'] = self.attr['opt_options']['SCIPY-BFGS']['gtol']
        init_dict['SCIPY-BFGS']['eps'] = self.attr['opt_options']['SCIPY-BFGS']['eps']

        init_dict['SCIPY-POWELL'] = dict()
        init_dict['SCIPY-POWELL']['xtol'] = self.attr['opt_options']['SCIPY-POWELL']['xtol']
        init_dict['SCIPY-POWELL']['ftol'] = self.attr['opt_options']['SCIPY-POWELL']['ftol']

        # 6) Cutoffs
        init_dict['CUTOFFS'] = self.attr['cutoffs']

        # 7) Questions
        for q in questions:
            init_dict['QUESTIONS'][q] = paras_obj.get_para(q)

        # 8) Preference parameters
        if version in ['scaled_archimedean']:
            init_dict['UNIATTRIBUTE SELF']['marginal'] = self.attr['marginals'][0]
            init_dict['UNIATTRIBUTE SELF']['r'] = paras_obj.get_para('r_self')
            init_dict['UNIATTRIBUTE SELF']['max'] = self.attr['upper'][0]

            init_dict['UNIATTRIBUTE OTHER']['marginal'] = self.attr['marginals'][1]
            init_dict['UNIATTRIBUTE OTHER']['r'] = paras_obj.get_para('r_other')
            init_dict['UNIATTRIBUTE OTHER']['max'] = self.attr['upper'][1]

            init_dict['MULTIATTRIBUTE COPULA']['delta'] = paras_obj.get_para('delta')
            init_dict['MULTIATTRIBUTE COPULA']['self'] = paras_obj.get_para('self')
            init_dict['MULTIATTRIBUTE COPULA']['other'] = paras_obj.get_para('other')
        elif version in ['nonstationary']:
            init_dict['ATEMPORAL']['alpha'] = paras_obj.get_para('alpha')
            init_dict['ATEMPORAL']['beta'] = paras_obj.get_para('beta')
            init_dict['ATEMPORAL']['gamma'] = paras_obj.get_para('gamma')
            init_dict['ATEMPORAL']['y_scale'] = paras_obj.get_para('y_scale')

            init_dict['DISCOUNTING']['discount_factors_0'] = \
                paras_obj.get_para('discount_factors_0')
            init_dict['DISCOUNTING']['discount_factors_1'] = \
                paras_obj.get_para('discount_factors_1')
            init_dict['DISCOUNTING']['discount_factors_3'] = \
                paras_obj.get_para('discount_factors_3')
            init_dict['DISCOUNTING']['discount_factors_6'] = \
                paras_obj.get_para('discount_factors_6')
            init_dict['DISCOUNTING']['discount_factors_12'] = \
                paras_obj.get_para('discount_factors_12')
            init_dict['DISCOUNTING']['discount_factors_24'] = \
                paras_obj.get_para('discount_factors_24')

            init_dict['DISCOUNTING']['unrestricted_weights_0'] = \
                paras_obj.get_para('unrestricted_weights_0')
            init_dict['DISCOUNTING']['unrestricted_weights_1'] = \
                paras_obj.get_para('unrestricted_weights_1')
            init_dict['DISCOUNTING']['unrestricted_weights_3'] = \
                paras_obj.get_para('unrestricted_weights_3')
            init_dict['DISCOUNTING']['unrestricted_weights_6'] = \
                paras_obj.get_para('unrestricted_weights_6')
            init_dict['DISCOUNTING']['unrestricted_weights_12'] = \
                paras_obj.get_para('unrestricted_weights_12')
            init_dict['DISCOUNTING']['unrestricted_weights_24'] = \
                paras_obj.get_para('unrestricted_weights_24')
        else:
            raise TrempyError('version not implemented')

        print_init_dict(init_dict, fname)

    def _check_integrity(self):
        """Check the integrity of the class instance."""
        # Distribute class attributes for further processing.
        args = ['paras_obj', 'sim_seed', 'sim_agents', 'sim_file', 'est_agents', 'maxfun',
                'est_file', 'questions', 'start', 'num_skip']

        paras_obj, sim_seed, sim_agents, sim_file, est_agents, maxfun, est_file, questions, \
            start, num_skip = dist_class_attributes(self, *args)

        # We restrict the identifiers for the questions between 1 and 16
        np.testing.assert_equal(12 < min(questions) <= max(questions) < 46, True)

        # The number of skipped individuals has to be non-negative.
        np.testing.assert_equal(0 <= num_skip, True)

        # We have to alternative how to start the estimation.
        np.testing.assert_equal(start in ['init', 'auto'], True)
