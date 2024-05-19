import os
import json
from env import JSSPEnv
import torch
import copy
import numpy as np
import random

def read_files(dir_path = "data/train"):
    # Store filenames from the specified directory path
    name_files = []
    for file_path in os.listdir(dir_path):
        full_path = os.path.join(dir_path, file_path)
        if os.path.isfile(full_path):
            name_files.append(full_path)
    return name_files

def generate_expert_obs(name_files, max_operations = 2, mult = 0.5):
    expert_observations = []
    num_actions = []

    for r in name_files:
        with open(r, 'r') as json_file:
            data = json.load(json_file)
        
        num_machines = len(data["operations"][0])
        job_advance = [0]*len(data["jobs"])  # Tracks the advancement of jobs
        num_steps = 0
        # Sort steps by machine and then by start and end time to process in order
        sorted_steps = sorted(data["result"], key=lambda k: (k["machine"], k["start"], k["end"]))
        
        # Initialize start times for jobs and machines if not already present in data
        if "jobs_starts" not in data:
            data["jobs_starts"] = [0]*len(data["jobs"])
            data["machines_starts"] = [0]*len(data["operations"][0])
        
        # Set up the environment with the data and configuration parameters
        env = JSSPEnv([data], max_operations, data["jobs_starts"], data["machines_starts"])
        obs = env.reset()
        
        # Main loop for processing operations until all are complete or a stopping condition is met
        while True:
            options = obs[('machine', 'exec', 'job')]
            delete_indexes = []
            action = [None]*num_machines
            selected_jobs = []
            selected_machines = []
            
            # Determine the next action for each machine based on sorted job steps
            for m in range(num_machines):
                flag = False
                for i, s in enumerate(sorted_steps):
                    if int(s["machine"]) == m:
                        selected_machines.append(m)
                        flag = True
                        # Check if the current job can proceed with its next task
                        if job_advance[int(s["job_id"])] == s["task_id"] and int(s["job_id"]) not in selected_jobs:
                            num_steps += 1
                            action[m] = int(s["job_id"])
                            delete_indexes.append(i)
                            job_advance[int(s["job_id"])] += 1
                            selected_jobs.append(int(s["job_id"]))
                            break
            
            # Create an auxiliary tensor to record the actions taken
            aux = torch.zeros(obs['machine', 'exec', 'job'].edge_index.shape[1])
            for o_i in range(len(options["edge_index"][0])):
                o = options["edge_index"][:, o_i]
                if action[o[0]] == int(o[1]):
                    aux[o_i] = 1
            
            # Check if all actions are completed for this step
            if obs['machine', 'exec', 'job'].edge_index.shape[1] == torch.sum(aux == 1).item():
                break
            
            # Normalize and update the observation state, then append to expert observations
            if (sum(aux) == 0 or obs['machine', 'exec', 'job'].edge_index.shape[1] < 2) == False:
                new_obs = env.normalize_state(obs)
                new_obs['machine', 'exec', 'job'].y = copy.deepcopy(aux)
                expert_observations.append(copy.deepcopy(new_obs))
                num_actions.append(int(obs['machine', 'exec', 'job'].edge_index.shape[1]))

            # Update the list of steps to be processed
            sorted_steps = [sorted_steps[i] for i in range(len(sorted_steps)) if i not in delete_indexes]
            obs, _, done, _ = env.step(action)

            # Break the loop if all operations are processed
            if num_steps >= len(data["operations"]):
                break

    # Sample a subset of observations to reduce dataset size and speed up training
    expert_observations = [expert_observations[x] for x in np.argsort(num_actions)]
    aux = random.sample(range(len(expert_observations)), int(len(expert_observations)*mult))
    aux.sort()
    expert_observations = [expert_observations[i] for i in aux]

    return expert_observations
