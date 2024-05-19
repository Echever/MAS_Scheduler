import copy
import gym
import numpy as np
import random
from torch_geometric.data import HeteroData
import torch
import heapq


class JSSPEnv(gym.Env):

    def __init__(self, instances, max_operations = 1, jobs_starts = None, machines_starts = None):
        super(JSSPEnv, self).__init__()
        self.instances = instances
        self.current_instance = 0
        self.max_operations = max_operations

        self.num_features_job = 4
        self.num_features_oper = 6
        self.num_features_mach = 8

        self.jobs_starts = jobs_starts
        self.machines_starts = machines_starts

    def generate_instance(self):

        self.displayed_ops = 0

        for i, v in enumerate(self.pending_ops_per_job):
            self.displayed_ops+=min(self.max_operations, v)
            self.display_ops_per_job[i] = min(self.max_operations, v)

        self.data = HeteroData()

        self.data["job"].x = torch.zeros((self.num_jobs, self.num_features_job), dtype= torch.float)
        self.data["operation"].x = torch.zeros((self.displayed_ops, self.num_features_oper), dtype= torch.float)
        self.data["machine"].x = torch.zeros((self.num_machines, self.num_features_mach), dtype= torch.float)
        
        current_op = 0
        aux_list = []
        for i in range(self.num_jobs):
            for _ in range(self.display_ops_per_job[i]):
                aux_list.append([current_op, i])
                current_op+=1

        self.data['operation', 'belongs', 'job'].edge_index = torch.LongTensor(aux_list).T

        current_op = 0
        aux_list = []
        for i, j in enumerate(self.jobs):
            if self.advance_operations[i] != len(j):
                current_op+=1
                aux_list.append([current_op-1, current_op-1])
                for _ in range(self.display_ops_per_job[i]-1):
                    aux_list.append([current_op, current_op-1]) # Comprobar
                    current_op+=1
            else:
                self.data["job"].x[i,0]=1

        self.data['operation', 'prec', 'operation'].edge_index = torch.LongTensor(aux_list).T

        aux_list = []
        for i in range(self.num_machines):
            for j in range(self.num_machines):
                if i!=j:
                    aux_list.append([i, j])
        self.data['machine', 'listens', 'machine'].edge_index = torch.LongTensor(aux_list).T

        current_pendings = []
        aux_list = []
        for i in range(self.num_jobs):
            try:
                if self.advance_operations[i] != len(self.jobs[i]):
                    current_pendings.append(self.all_pendings[self.jobs[i][self.advance_operations[i]]])
            except:
                pass
            if self.display_ops_per_job[i] == 0:
                continue
            for j in range(self.num_jobs):
                if self.display_ops_per_job[j] == 0:
                    continue
                if i!=j or True:
                    aux_list.append([i, j])

        self.data['job', 'listens', 'job'].edge_index = torch.LongTensor(aux_list).T

        aux_list = []
        aux_list_2 = []
        aux_list_features2 = []

        current_op = 0
        for i in range(self.num_jobs):
            for v in range(self.display_ops_per_job[i]):
                o = self.operations[self.jobs[i][v + self.advance_operations[i]]]
                for j in range(len(o)):
                    t = o[j]
                    if t!=0:
                        aux_list.append([current_op, j])
                        aux_list_2.append([j, current_op])
                        aux_list_features2.append([t, t/np.sum(o), t/self.all_pendings[self.advance_operations[i]], 0])
                current_op+=1

        self.data['operation', 'exec', 'machine'].edge_index = torch.LongTensor(aux_list).T
        self.data['operation', 'exec', 'machine'].edge_attr = torch.Tensor(aux_list_features2)

        self.data['machine', 'exec', 'operation'].edge_index = torch.LongTensor(aux_list_2).T
        self.data['machine', 'exec', 'operation'].edge_attr = torch.Tensor(aux_list_features2)

        current_op = 0
        self.aval_machines = np.array([False]*self.num_machines)
        for i in range(self.num_jobs):
            o_index = 0
            for v in range(self.display_ops_per_job[i]):
                self.data["operation"].x[current_op, 1] = self.all_pendings[self.jobs[i][v + self.advance_operations[i]]]
                aux = np.array(self.operations[self.jobs[i][v + self.advance_operations[i]]])
                self.aval_machines[aux!=0] = True
                self.data["operation"].x[current_op, 2] = np.min(aux[np.array(aux)!=0])
                self.data["operation"].x[current_op, 3] = np.sum(aux[np.array(aux)!=0])/self.all_pendings[self.jobs[i][v+ self.advance_operations[i]]]
                self.data["operation"].x[current_op, 4] = len(self.jobs[i]) - o_index - self.advance_operations[i]
                self.data["operation"].x[current_op, 5] = self.min_pendings[self.jobs[i][v + self.advance_operations[i]]]
                o_index+=1
                current_op+=1

        for i, v in enumerate(self.aval_machines):
            if not v:
                continue
            self.data["machine"].x[i,0] =self.machines_final_time[i]
            if self.num_steps != 0:
                self.data["machine"].x[i,1] = self.machines_occupations[i] / (self.data["machine"].x[i,0] + 0.00001)
            self.data["machine"].x[i,2] = self.data["machine"].x[i,0] - torch.min(self.data["machine"].x[i,0])

        aux_list = []
        aux_list_2 = []
        aux_list_features = []

        min_starts = []
        for j_id in range(len(self.jobs)):
            if self.advance_operations[j_id] != len(self.jobs[j_id]):
                oper = self.operations[self.current_operations[j_id]]
                for m in range(len(oper)):
                    t = oper[m]
                    if t!=0:
                        min_starts.append(max(self.operations_ends[j_id], self.data["machine"].x[m, 0]))
                        break

        self.best_starts_min = min(min_starts)
        self.best_starts_min_k = (heapq.nsmallest(int(self.num_machines), min_starts) if min_starts else [])[-1]

        self.final_value = None

        self.min_starts = np.array(min_starts)

        self.mult_per = 1.05
        self.min_options = 2

        if sum(self.min_starts < float(self.best_starts_min*self.mult_per)) > self.num_machines:
            self.final_value =  self.best_starts_min_k
        elif sum(self.min_starts < float(self.best_starts_min*self.mult_per)) < self.min_options:
            self.final_value = (heapq.nsmallest(self.min_options, min_starts) if min_starts else [])[-1]
        else:
            self.final_value = self.best_starts_min*self.mult_per


        total_gap = 0

        self.machine_candidates = {}
        counter_job = 0

        self.final_consideration = []
        for j_id in range(len(self.jobs)):
            if self.advance_operations[j_id] != len(self.jobs[j_id]):
                oper = self.operations[self.current_operations[j_id]]
                for m in range(len(oper)):
                    t = oper[m]
                    if t!=0:
                        if float(self.final_value) < float(max(self.operations_ends[j_id], self.data["machine"].x[m, 0])) :
                            self.final_consideration.append(False)
                        else:
                            self.final_consideration.append(True)

                        if m not in self.machine_candidates:
                            self.machine_candidates[m] = []

                        self.machine_candidates[m].append([j_id, self.advance_operations[j_id] == len(self.jobs[j_id])-1, counter_job])
                        counter_job+=1

                        calcu = t + max(self.operations_ends[j_id] - self.data["machine"].x[m, 0],0)
                        total_gap += calcu
                        aux_list.append([m, j_id])
                        aux_list_2.append([j_id, m])
                        aux_list_features.append([calcu, t/np.sum(oper) , t + max(self.operations_ends[j_id], self.data["machine"].x[m, 0]) ])

                        break



        for l in aux_list_features:
            l[1] = (l[0]/total_gap)
            l.append(0)

        aux_list = torch.LongTensor(aux_list).T
        aux_list_2 = torch.LongTensor(aux_list_2).T
        aux_list_features = torch.Tensor(aux_list_features)

        if all([not f for f in self.final_consideration]):
            assert True == False

        self.data['machine', 'exec', 'job'].edge_index = aux_list
        self.data['machine', 'exec', 'job'].edge_attr = aux_list_features
        self.data['machine', 'exec', 'job'].final_consideration = torch.BoolTensor(self.final_consideration)

        self.data['job', 'exec', 'machine'].edge_index = aux_list_2
        self.data['job', 'exec', 'machine'].edge_attr = aux_list_features

        self.state = copy.deepcopy(self.data)
        self.calculate_next_state()

    def reset(self, sel_index = None):

        if sel_index is None:
            instance = self.instances[self.current_instance]
            self.current_instance = (self.current_instance + 1)%len(self.instances)
        else:
            instance = self.instances[sel_index]

        jobs, operations = instance["jobs"], instance["operations"]

        self.jobs = jobs
        self.num_jobs = len(jobs)
        self.operations = operations
        self.num_operations = len(operations)
        self.num_machines = len(instance["operations"][0])
        self.total_operations = self.num_operations 
        
        if self.jobs_starts is not None:
            self.operations_ends = self.jobs_starts
        else:
            self.operations_ends = [0]*self.num_jobs


        self.advance_operations = [0]*self.num_jobs
        self.machines_occupations = [0]*self.num_machines

        if self.machines_starts is not None:
            self.machines_final_time = self.machines_starts
        else:
            self.machines_final_time = [0]*self.num_machines
        self.current_operations = [0]*self.num_jobs

        self.pending_ops_per_job = [len(j) for j in self.jobs]
        self.display_ops_per_job = [0]*self.num_jobs
        self.current_op_per_job = [0]*self.num_jobs

        for j_id in range(len(self.jobs)):
            self.current_operations[j_id] = self.jobs[j_id][0]
        self.num_steps = 0

        self.all_pendings  = []
        self.min_pendings  = []

        for job in jobs:
            aux = []
            aux_min = []
            for o_id in reversed(job):
                aux_o = np.array(self.operations[o_id])
                try:
                    aux_min.append(np.min(aux_o[np.where(aux_o!=0)]))
                    aux.append(np.mean(aux_o[np.where(aux_o!=0)]) + aux[-1])
                except:
                    aux.append(np.mean(aux_o[np.where(aux_o!=0)]))
                    aux_min.append(np.min(aux_o[np.where(aux_o!=0)]))

            self.all_pendings = self.all_pendings + list(reversed(aux))
            self.min_pendings = self.min_pendings + list(reversed(aux_min))
            

        self.generate_instance()

        return self.state
    
    def calculate_next_state(self):
        
        for j_id in range(len(self.jobs)):
            if int(self.state["job"].x[j_id,0])==0:
                o_id = self.current_operations[j_id]
                pj = self.state['operation', 'belongs', 'job'].edge_index[:,self.state['operation', 'belongs', 'job'].edge_index[1,:] == j_id]
                oper_id = pj[0,0]
                self.state["operation"].x[oper_id, 0] = 1
                self.state["operation"].x[oper_id, 1] = self.all_pendings[o_id]
                self.state["job"].x[j_id, 1] = self.operations_ends[j_id]
                self.state["job"].x[j_id, 2] =  len(self.jobs[j_id]) - self.advance_operations[j_id]
                self.state["job"].x[j_id, 3] = self.all_pendings[o_id]

                aux_ind  = 0
                prev_end = self.operations_ends[j_id]
                for opers in pj.T:
                    self.state["operation"].x[opers[0], 5] = self.min_pendings[o_id + aux_ind] + prev_end
                    prev_end = self.min_pendings[o_id + aux_ind] + prev_end
                    aux_ind+=1

        for m in range(len(self.state["machine"].x)):
            mask = self.state["operation", "exec", "machine"].edge_index[1,:] == m
            if mask.any().item():
                self.state["operation", "exec", "machine"].edge_attr[mask,3] = self.state["operation", "exec", "machine"].edge_attr[mask,0]/self.state["operation", "exec", "machine"].edge_attr[mask,0].max()
                mask = self.state["machine", "exec", "operation"].edge_index[0,:] == m
                self.state["machine", "exec", "operation"].edge_attr[mask,3] = self.state["machine", "exec", "operation"].edge_attr[mask,0]/self.state["machine", "exec", "operation"].edge_attr[mask,0].max()
                self.state["machine"].x[m, 3] = self.state["operation", "exec", "machine"].edge_attr[mask,0].max()
                self.state["machine"].x[m, 4] = self.state["operation", "exec", "machine"].edge_attr[mask,0].min()
                self.state["machine"].x[m, 5] = self.state["operation", "exec", "machine"].edge_attr[mask,0].mean()
                self.state["machine"].x[m, 6] = self.state["operation", "exec", "machine"].edge_attr[mask,0].shape[0] # Cambiar para que tenga una imagen más global
            else:
                self.state["machine"].x[m, 3] = 0
                self.state["machine"].x[m, 4] = 0
                self.state["machine"].x[m, 5] = 0
                self.state["machine"].x[m, 6] = 0

                self.state['machine', 'listens', 'machine'].edge_index = self.state['machine', 'listens', 'machine'].edge_index[:, self.state['machine', 'listens', 'machine'].edge_index[0,:] != m]
                self.state['machine', 'listens', 'machine'].edge_index = self.state['machine', 'listens', 'machine'].edge_index[:, self.state['machine', 'listens', 'machine'].edge_index[1,:] != m]

            mask = self.state["machine", "exec", "job"].edge_index[0,:] == m

            if mask.any().item():
                self.state["machine", "exec", "job"].edge_attr[mask,3] = self.state["machine", "exec", "job"].edge_attr[mask,0]/self.state["machine", "exec", "job"].edge_attr[mask,0].max()
                self.state["job", "exec", "machine"].edge_attr[mask,3] = self.state["job", "exec", "machine"].edge_attr[mask,0]/self.state["job", "exec", "machine"].edge_attr[mask,0].max()
                self.state["machine"].x[m, 7] = self.state["machine", "exec", "job"].edge_attr[mask,0].shape[0]
            else:
                self.state["machine"].x[m, 7] = 0

    def step(self, action):

        for sel_mach, sel_job in enumerate(action):
            if sel_job is None:
                continue 

            self.num_steps+=1       

            self.advance_operations[sel_job]+=1
            self.pending_ops_per_job[sel_job]-=1
            
            prev_ms = float(torch.max(self.state["machine"].x[:,0]))
            o_id = self.current_operations[sel_job]
            start_time = max(self.state["machine"].x[sel_mach,0], self.operations_ends[sel_job])
            proc_time  = self.operations[o_id][sel_mach]

            final_time =  start_time + proc_time
            self.machines_final_time[sel_mach] = float(final_time)
            self.operations_ends[sel_job] = final_time
            self.machines_occupations[sel_mach] += proc_time
            
            reward = prev_ms - float(torch.max(self.state["machine"].x[:,0])) 
            self.current_operations[sel_job]+=1
            
            if self.total_operations == self.num_steps:
                self.mk = round(float(max(self.machines_final_time)),2)
                return self.state, reward , True, {}
            
            done = False
            total_reward = reward

        self.generate_instance()
        return self.state, total_reward , done, {}

    def sample(self):
        action = [None]*self.num_machines
        for m_id in range(self.num_machines):
            mask = self.state['machine', 'exec', 'job']["edge_index"][0] == m_id
            if mask.any():
                opts = self.state['machine', 'exec', 'job']["edge_index"][1][mask]
                final_opt = random.choice(opts)              
                if final_opt not in action:
                    action[m_id] = final_opt
        return action
    
    def normalize_state(self, state):
        state = copy.deepcopy(state)
        for i in range(state["job"].x.shape[1]):
            state["job"].x[:,i] = (2*(state["job"].x[:,i] - state["job"].x[:,i].min())/(state["job"].x[:,i].max() - state["job"].x[:,i].min() + 1e-7 )-1).float()
        
        for i in range(state["operation"].x.shape[1]):
            state["operation"].x[:,i] = (2*(state["operation"].x[:,i] - state["operation"].x[:,i].min())/(state["operation"].x[:,i].max() - state["operation"].x[:,i].min() + 1e-7 )-1).float()
        
        for i in range(state["machine"].x.shape[1]):
            state["machine"].x[:,i] = (2*(state["machine"].x[:,i] - state["machine"].x[:,i].min())/(state["machine"].x[:,i].max() - state["machine"].x[:,i].min() + 1e-7 )-1).float()

        state[('operation', 'exec', 'machine')].edge_attr = (2*(state[('operation', 'exec', 'machine')].edge_attr -  state[('operation', 'exec', 'machine')].edge_attr.min())/(state[('operation', 'exec', 'machine')].edge_attr.max() - state[('operation', 'exec', 'machine')].edge_attr.min() + 1e-7 )-1).float()
        state[('machine', 'exec', 'operation')].edge_attr = (2*(state[('machine', 'exec', 'operation')].edge_attr -  state[('machine', 'exec', 'operation')].edge_attr.min())/(state[('machine', 'exec', 'operation')].edge_attr.max() - state['machine', 'exec', 'operation'].edge_attr.min() + 1e-7 )-1).float()
        state[('machine', 'exec', 'job')].edge_attr = (2*(state[('machine', 'exec', 'job')].edge_attr - state[('machine', 'exec', 'job')].edge_attr.min())/(state[('machine', 'exec', 'job')].edge_attr.max() - state[('machine', 'exec', 'job')].edge_attr.min() + 1e-7 )-1).float()
        state[('job', 'exec', 'machine')].edge_attr = (2*(state[('job', 'exec', 'machine')].edge_attr - state[('job', 'exec', 'machine')].edge_attr.min())/(state[('job', 'exec', 'machine')].edge_attr.max() - state[('job', 'exec', 'machine')].edge_attr.min() + 1e-7 )-1).float()
        return state
    
