"""Estimate the model agent-by-agent."""

import shutil
import shlex
import copy
import os

import pandas as pd
import numpy as np

from trempy.shared.shared_auxiliary import print_init_dict
from trempy.tests.test_auxiliary import get_rmse
from trempy.read.read import single_agent_init
from trempy.estimate.estimate import estimate
from trempy.read.read import read


def individual_estimation(fname):
    """Loop over all agents and estimate the model each time."""
    individual_estimate_cleanup()

    # Get start dictionary
    init_dict = read(fname)
    init_dict['']

    num_agents = init_dict['ESTIMATION']['agents']

    os.mkdir('agents_out')
    os.chdir('agents_out')

    print('Total number of agents: {}.'.format(num_agents))

    for agent in np.arange(0, num_agents):
        print('Current agent: {}.'.format(agent))

        # Create a new directory for each agent
        os.mkdir('agent_{}'.format(agent))
        os.chdir('agent_{}'.format(agent))

        init_agent = copy.deepcopy(init_dict)
        single_agent_init(init_agent)

        # Select the nth agent.
        init_agent['ESTIMATION']['skip'] = agent
        init_agent['ESTIMATION']['agents'] = 1
        init_agent['SIMULATION']['agents'] = 1

        # Update filepath
        init_agent['ESTIMATION']['file'] = '../../' + init_agent['ESTIMATION']['file']
        init_agent['SIMULATION']['file'] = '../../' + init_agent['SIMULATION']['file']

        # Save the current dictionary.
        agent_fname = 'agent_{}.trempy.ini'.format(agent)
        print_init_dict(init_agent, fname=agent_fname)

        # Estimate model for nth agent.
        estimate(agent_fname)

        os.chdir('..')

    # Back at the original level of the init file we started with.
    os.chdir('..')


def collect_parameters(agents):
    """Collect individual-level estimates in one dataframe."""
    df = list()
    for agent in np.arange(0, agents):
        print(agent)
        # Start values
        start_dict = read('agents_out/agent_{}/start/start.trempy.ini'.format(agent))
        atemporal_paras = start_dict['ATEMPORAL']
        discounting_paras = start_dict['DISCOUNTING']
        std = start_dict['QUESTIONS']

        start_values = merge_two_dicts(atemporal_paras, discounting_paras)
        start_values['sd_temporal'] = std[1]
        start_values['sd_risk'] = std[2]
        start_values = {key + '_start': value for key, value in start_values.items()}

        # Stop values
        stop_dict = read('agents_out/agent_{}/stop/stop.trempy.ini'.format(agent))
        atemporal_paras = stop_dict['ATEMPORAL']
        discounting_paras = stop_dict['DISCOUNTING']
        std = stop_dict['QUESTIONS']

        stop_values = merge_two_dicts(atemporal_paras, discounting_paras)
        stop_values['sd_temporal'] = std[1]
        stop_values['sd_risk'] = std[2]
        stop_values = {key + '_end': value for key, value in start_values.items()}

        # merge
        agent_result = merge_two_dicts(start_values, stop_values)

        # Keep parameter estimates only, ignore bounds and "fixed".
        agent_result = {key: value[0] for key, value in agent_result.items()}

        # Agent ID
        agent_result['agent_number'] = agent

        # Add diagnostic information
        os.chdir('agents_out/agent_{}'.format(agent))
        agent_result['rmse'] = round(get_rmse(), 6)
        agent_result['participant_id'] = get_participant_id()

        num_steps, num_evals, terminated, success, message, crit_val_start, crit_val_end = \
            get_diagnostics()
        agent_result['num_steps'] = num_steps
        agent_result['num_evals'] = num_evals
        agent_result['terminated'] = terminated
        agent_result['success'] = success
        agent_result['message'] = message
        agent_result['crit_val_start'] = crit_val_start
        agent_result['crit_val_end'] = crit_val_end

        os.chdir('../..')

        df.append(agent_result)

    # convert list of dictionaries to dataframe.
    output = pd.DataFrame(df)
    return output


def individual_estimate_cleanup():
    """Ensure that we start the estimation with a clean slate."""
    # We remove the directories that contain the simulated choice menus at the start.
    for dirname in ['agents_out']:
        if os.path.exists(dirname):
            shutil.rmtree(dirname)

    # We remove the information from earlier estimation runs.
    for fname in ['est.trempy.info', 'est.trempy.log', '.stop.trempy.scratch']:
        if os.path.exists(fname):
            os.remove(fname)


def get_participant_id():
    """Return the participant ID from the information file."""
    with open('compare.trempy.info') as in_file:
        for line in in_file.readlines():
            if 'Individual' in line:
                stat = shlex.split(line)[1]
                if stat not in ['---']:
                    stat = float(stat)
                return stat


def get_diagnostics():
    """Return diagnostics for the individual's estimation."""
    with open('est.trempy.info') as in_file:
        num_steps = 0
        num_evals = 0
        terminated = False

        lines = in_file.readlines()
        for index, line in enumerate(lines):
            if 'Number of Evaluations' in line:
                stat = shlex.split(line)[3]
                if stat not in ['---']:
                    num_evals = float(stat)
            if 'Number of Steps' in line:
                stat = shlex.split(line)[3]
                if stat not in ['---']:
                    num_steps = float(stat)

            if 'TERMINATED' in line:
                terminated = True

            # Criterion function value at the end
            condition = ('Identifier' not in line) and ('Questions' not in line) and \
                        ('Start' in line) and ('Step' in line)
            if condition:
                criterion_line = lines[index + 2]
                crit_val_start, crit_val_end = shlex.split(criterion_line)[:2]
                crit_val_start = round(float(crit_val_start), 6)
                crit_val_end = round(float(crit_val_end), 6)

    with open('est.trempy.log') as in_file:
        success = False
        for line in in_file.readlines():

            if 'Message' in line:
                message = shlex.split(line)[1:]
                message = " ".join(message)
            if 'Success' in line:
                stat = shlex.split(line)[1]
                if stat in ['False']:
                    success = False
                if stat in ['True']:
                    success = True

    return num_steps, num_evals, terminated, success, message, crit_val_start, crit_val_end


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z
