from env import JSSPEnv
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from parsedata import get_file_data
import torch
from torch_geometric.loader import DataLoader
import random
import pandas as pd
from model import Model
from generate_expert_obs import read_files

device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()

# Function to apply padding to a list of tensors for consistent tensor dimensions
def apply_padding(tensor_list):
    padding = int(max([i.size() for i in tensor_list])[0])
    return [
        F.pad(tensor, (0, padding - tensor.size()[0]), mode="constant", value=0)
        for tensor in tensor_list
    ]

# Calculate accuracy by comparing predicted actions against true targets
def get_accuracy(list_action_prob, list_targets):
    acc = 0
    all = 0
    for i in range(len(list_action_prob)):
        for j, value in enumerate(list_action_prob[i]):
            aux = 0
            all+=1
            if list_action_prob[i][j]>0:
                aux = 1
            if list_targets[i][j] == aux:
                acc+=1
    acc = acc / all
    return acc

# Evaluate model performance on a validation dataset
def evaluate_val_folder(folder, model, benchmark, max_operations):
    val_name_files = read_files(folder)
    final_res = []
    for i in range(len(val_name_files)):
        res2 , _, opt= evaluate_instance(model, benchmark, max_operations, sel_instance = i, dir_path=folder, name_files = val_name_files)
        final_res.append((res2 - opt)/opt)

    final_per = np.mean(final_res)
    return final_per

# Evaluate a specific instance from the dataset using the trained model
def evaluate_instance(model, benchmark = False, max_operations = 2, sel_instance = 0, dir_path = "taillard", name_files = []):
        jobs_starts = None
        machines_starts = None
        if benchmark:
            onlyfiles = []
            dir_path = dir_path
            for path in os.listdir(dir_path):
                if os.path.isfile(os.path.join(dir_path, path)):
                    onlyfiles.append(path)

            for index, value in enumerate(onlyfiles[sel_instance:sel_instance+1]):
                info = get_file_data(value, dir_path)

                jobs, operations = info[0], info[1]
                instances = [{"jobs": jobs, "operations": operations}]
                num_machines = len(operations[0])
                opt = 0

        else:
            with open(name_files[sel_instance], 'r') as json_file: 
                    data = json.load(json_file) 
            
            num_machines = len(data["operations"][0])
            job_advance = [0]*len(data["jobs"])

            jobs, operations = data["jobs"], data["operations"]

            if "jobs_starts" not in data:
                data["jobs_starts"] = None
                data["machines_starts"] = None

            instances = [data]
            opt = data["score"]
            jobs_starts, machines_starts =  data["jobs_starts"], data["machines_starts"]

        env = JSSPEnv(instances, max_operations, jobs_starts, machines_starts)
        obs = env.reset()
        operations_counter = 0
        model = model.to(device)
        
        counter_inferences = 0
        while True:
            with torch.no_grad():
                nobs = env.normalize_state(obs)
                nobs = nobs.to(device)
                res = model(nobs)
                res = res.T[0]
                counter_inferences +=1
                res = res - res.mean()
                res = res.cpu()
                nobs = nobs.cpu()
                minvalue = 0                          

                if res.max()<0:
                    minvalue = res.max()

                action = [None]*num_machines
                action_values = [None]*num_machines
                action_counter = 0
                for m_i in range(num_machines):
                    mask = (nobs['machine', 'exec', 'job']['edge_index'][0][nobs['machine', 'exec', 'job'].final_consideration] == m_i)
                    if mask.any():
                        mvalue = res[mask].max()
                        if mvalue >= minvalue:
                            indexes = (res == mvalue).nonzero(as_tuple=True)
                            final_indexes = []
                            if indexes[0].shape[0] > 1:
                                for ind in list(indexes[0]):
                                    if int(nobs['machine', 'exec', 'job']['edge_index'][0][nobs['machine', 'exec', 'job'].final_consideration][ind]) == m_i:
                                        final_indexes.append(ind)
                                final_indexes = final_indexes[0]
                            else:
                                final_indexes = indexes[0]
                                                                
                            action[m_i] = int(nobs['machine', 'exec', 'job']['edge_index'][1][nobs['machine', 'exec', 'job'].final_consideration][final_indexes])
                            action_values[m_i] = mvalue
                            action_counter+=1
                            operations_counter+=1

                obs, _, done, _ = env.step(action)
                if done:
                    break
        
        return max(env.machines_final_time), counter_inferences, opt

# Function to train the model on given data
def train(model, optimizer, criterion, train_loader):
    model.train()
    total_examples, total_loss, total_acc = 0, 0, 0

    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        res = model(batch)

        list_action_prob = []
        list_targets = []

        row, col = batch[('machine', 'exec', 'job')].edge_index
        batch_index = batch["machine"].batch[row][batch[('machine', 'exec', 'job')].final_consideration]

        for i in range(batch["operation"].batch[-1] + 1):
            action_probs = res[batch_index==i].T[0]
            action_probs = action_probs - action_probs.mean()

            list_action_prob.append(action_probs)
            list_targets.append(batch[('machine','exec','job')].y[batch[('machine','exec','job')].final_consideration][batch_index==i])


        acc = get_accuracy(list_action_prob, list_targets)

        list_action_prob = torch.stack(apply_padding(list_action_prob))
        list_targets = torch.stack(apply_padding(list_targets))
        loss = criterion(list_action_prob, list_targets)
            
        loss.backward()
        optimizer.step()
        len_bach_size = len(batch)
        total_examples += len_bach_size
        total_loss += float(loss) * len_bach_size
        total_acc += float(acc) * len_bach_size

    return total_loss / total_examples, total_acc / total_examples

    
def start_train(expert_observations, max_operations, epochs = 40, learning_rate = 0.0003, batch_size = 64, hidden_channels = 64, num_layers = 2, heads = 2):

    epochs = epochs
    learning_rate = learning_rate
    batch_size = batch_size

    train_loader = DataLoader(expert_observations, batch_size=batch_size)

    model = Model(hidden_channels, num_layers= num_layers, heads= heads,  metadata= expert_observations[0].metadata())
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_result = 10
    for epoch in range(1, epochs):
        loss, acc = train(model, optimizer, criterion, train_loader)        
        print(epoch, round(acc,3), round(loss,3))
        if epoch>0:
            val_result = evaluate_val_folder("data/val", model, False, max_operations)
            if val_result< best_val_result:
                best_val_result = val_result
                model_path = f"./model/model_" + str(round(best_val_result,4))+ ".pt"
                torch.save(model.state_dict(), model_path)
                print("best_val_result ", best_val_result)


    return val_result
