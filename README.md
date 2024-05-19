# Multi-Assignment Scheduler: A New Behavioral Cloning Method for the Job-Shop Scheduling Problem

This repository hosts the code for the paper "Multi-Assignment Scheduler: A New Behavioral Cloning Method for the Job-Shop Scheduling Problem," presented at the 18th Learning and Intelligent Optimization Conference (LION).

Recent advances in applying deep learning methods to complex scheduling problems have highlighted their potential in learning dispatching rules. While most studies have focused on deep reinforcement learning (DRL), this paper introduces a novel methodology aimed at learning dispatching policies for the job-shop scheduling problem (JSSP) by employing behavioral cloning and graph neural networks. By leveraging optimal solutions for the training phase, our approach sidesteps the need for exhaustive exploration of the solution space, thereby enhancing performance compared to DRL methods proposed in the literature. Additionally, we introduce a novel modeling of the JSSP with the aim of improving efficiency in terms of solving an instance in real time. This involves two key aspects: firstly, the creation of an action space that allows our policy to assign multiple operations to machines within a single action, substantially reducing the frequency of model usage; and secondly, the definition of a state space that only includes significant operations. We evaluated our methodology using a widely recognized open JSSP benchmark, comparing it against four state-of-the-art DRL methods and an enhanced metaheuristic approach, demonstrating superior performance.

## File Descriptions

Below is a brief description of each file within the repository:

- `env.py`: Contains the implementation of the environment class for the Job-Shop Scheduling Problem.

- `generate_expert_obs.py`: Responsible for generating expert observations from the provided data sets.

- `main.py`: The main script for running the model training process.

- `model.py`: Defines the neural network architecture.

- `parsedata.py`: Contains functions to parse and preprocess the input data files.

- `test.py`: Used to evaluate the trained model against test datasets.

- `train.py`: Contains the training loop and functions necessary for training the model.

- `transformer_conv.py`: Implements a transformer based GNN.

## Installation

Install all necessary Python packages with pip:
```
pip install -r requirements.txt
```