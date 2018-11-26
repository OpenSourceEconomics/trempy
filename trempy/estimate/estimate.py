#!/usr/bin/env python
"""Contains estimate function."""
import shutil
import copy

from scipy.optimize import minimize

from trempy.estimate.estimate_auxiliary import get_automatic_starting_values
from trempy.shared.shared_auxiliary import dist_class_attributes
from trempy.estimate.estimate_auxiliary import estimate_simulate
from trempy.estimate.estimate_auxiliary import estimate_cleanup
from trempy.estimate.clsEstimate import EstimateClass
from trempy.custom_exceptions import MaxfunError
from trempy.custom_exceptions import TrempyError
from trempy.process.process import process
from trempy.clsModel import ModelCls


def estimate(fname):
    """Estimate the model by the method of maximum likelihood."""
    estimate_cleanup()

    model_obj = ModelCls(fname)

    # Distribute class parameters except for economic parameters and version-specific thing
    args = [model_obj, 'version', 'est_file', 'questions', 'paras_obj', 'start', 'cutoffs',
            'maxfun', 'est_detailed', 'opt_options', 'optimizer', 'est_agents', 'num_skip']

    version, est_file, questions, paras_obj, start, cutoffs, maxfun, est_detailed, \
        opt_options, optimizer, est_agents, num_skip = dist_class_attributes(*args)

    # Handle version-specific objects not included in the para_obj
    if version in ['scaled_archimedean']:
        upper, marginals = dist_class_attributes(*[model_obj, 'upper', 'marginals'])
        version_specific = {'upper': upper, 'marginals': marginals}
    elif version in ['nonstationary']:
        version_specific = dict()

    # We only need to continue if there is at least one parameter to actually estimate.
    if len(paras_obj.get_values('optim', 'free')) == 0:
        raise TrempyError('no free parameter to estimate')

    # Some initial setup
    df_obs = process(est_file, questions, num_skip, est_agents, cutoffs)

    estimate_obj = EstimateClass(
        df=df_obs, cutoffs=cutoffs, questions=questions, paras_obj=copy.deepcopy(paras_obj),
        max_eval=maxfun, optimizer=optimizer, version=version, **version_specific)

    # We lock in an evaluation at the starting values as not all optimizers actually start there.
    if start in ['auto']:
        paras_obj = get_automatic_starting_values(
            paras_obj=paras_obj, df_obs=df_obs,
            questions=questions, version=version, **version_specific)

    # Objects for scipy.minimize
    x_optim_free_start = paras_obj.get_values('optim', 'free')
    x_free_bounds = paras_obj.get_bounds('free')
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
            bounds = None
        elif optimizer == 'SCIPY-POWELL':
            options['ftol'] = opt_options['SCIPY-POWELL']['ftol']
            options['xtol'] = opt_options['SCIPY-POWELL']['xtol']
            method = 'POWELL'
            bounds = None
        elif optimizer == 'SCIPY-L-BFGS-B':
            options['gtol'] = opt_options['SCIPY-L-BFGS-B']['gtol']
            options['ftol'] = opt_options['SCIPY-L-BFGS-B']['ftol']
            options['eps'] = opt_options['SCIPY-L-BFGS-B']['eps']
            method = 'L-BFGS-B'
            bounds = x_free_bounds
            # Add bounds
        else:
            raise TrempyError('flawed choice of optimization method')

        try:
            opt = minimize(estimate_obj.evaluate, x_optim_free_start, method=method,
                           options=options, bounds=bounds)
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
