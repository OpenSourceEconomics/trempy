#!/usr/bin/env python
import shutil
import copy

from scipy.optimize import minimize

from trempy.shared.shared_auxiliary import dist_class_attributes
from trempy.estimate.estimate_auxiliary import estimate_simulate
from trempy.estimate.estimate_auxiliary import estimate_cleanup
from trempy.estimate.clsEstimate import EstimateClass
from trempy.custom_exceptions import MaxfunError
from trempy.custom_exceptions import TrempyError
from trempy.estimate.estimate_auxiliary import get_automatic_starting_values
from trempy.process.process import process
from trempy.clsModel import ModelCls


def estimate(fname):
    """This function estimates the model by the method of maximum likelihood."""
    estimate_cleanup()

    model_obj = ModelCls(fname)

    est_file, questions, paras_obj, start, cutoffs, maxfun, est_detailed, opt_options, optimizer, est_agents = \
        dist_class_attributes(model_obj, 'est_file', 'questions', 'paras_obj', 'start',
            'cutoffs', 'maxfun', 'est_detailed', 'opt_options', 'optimizer', 'est_agents')

    # Some initial setup
    df_obs = process(est_file, questions, est_agents)

    estimate_obj = EstimateClass(df_obs, cutoffs, questions, copy.deepcopy(paras_obj), maxfun)

    # We lock in an evaluation at the starting values as not all optimizers actually start there.
    if start in ['auto']:
        paras_obj = get_automatic_starting_values(paras_obj, df_obs, questions)

    x_optim_free_start = paras_obj.get_values('optim', 'free')
    estimate_obj.evaluate(x_optim_free_start)

    # We simulate a sample at the starting point.
    if est_detailed:
        estimate_simulate('start', x_optim_free_start, model_obj, df_obs)

    # Optimization of likelihood function
    if maxfun > 1:

        options = dict()

        if optimizer == 'SCIPY-BFGS':
            options['gtol'] = opt_options['SCIPY-BFGS']['gtol']
            options['eps'] = opt_options['SCIPY-BFGS']['eps']
            method = 'BFGS'
        elif optimizer == 'SCIPY-POWELL':
            options['ftol'] = opt_options['SCIPY-POWELL']['ftol']
            options['xtol'] = opt_options['SCIPY-POWELL']['xtol']
            method = 'POWELL'
        else:
            raise TrempyError('flawed choice of optimization method')

        try:
            opt = minimize(estimate_obj.evaluate, x_optim_free_start, method=method,
                options=options)
        except MaxfunError:
            opt = dict()
            opt['message'] = 'Optimization reached maximum number of function evaluations.'
            opt['success'] = False
    else:
        # We are not faced with a serious estimation request.
        opt = dict()
        opt['message'] = 'Single evaluation of criterion function at starting values.'
        opt['success'] = False

    # Now we can wrap up all estimation related tasks.
    estimate_obj.finish(opt)

    # We simulate a sample at the stopping point.
    if est_detailed:
        x_econ_all_step = estimate_obj.get_attr('x_econ_all_step')
        paras_obj.set_values('econ', 'all', x_econ_all_step)
        x_optim_free_step = paras_obj.get_values('optim', 'free')
        estimate_simulate('stop', x_optim_free_step, model_obj, df_obs)
        shutil.copy('stop/compare.trempy.info', 'compare.trempy.info')

    # We only return the best value of the criterion function and the corresponding parameter
    # vector.
    rslt = list()
    rslt.append(estimate_obj.get_attr('f_step'))
    rslt.append(estimate_obj.get_attr('x_econ_all_step'))

    return rslt
