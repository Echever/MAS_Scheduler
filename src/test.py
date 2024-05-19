import argparse
from model import Model
import torch
import pandas as pd
from train import evaluate_instance


# Metadata for the model, specifies relationships and node types used in the graph neural network
metadata = (['job', 'operation', 'machine'],
            [('operation', 'belongs', 'job'),
            ('operation', 'prec', 'operation'),
            ('machine', 'listens', 'machine'),
            ('job', 'listens', 'job'),
            ('operation', 'exec', 'machine'),
            ('machine', 'exec', 'operation'),
            ('machine', 'exec', 'job'),
            ('job', 'exec', 'machine')])


def main(args):

    # Initialize the model with parameters from the command line
    model = Model(hidden_channels=args.hidden_channels, metadata=metadata, num_layers=args.num_layers, heads=args.heads)
    model.load_state_dict(torch.load(args.model_path))  # Load model weights

    # Read optimal solutions from a CSV file for comparison
    opts = list(pd.read_csv(f"{args.folder}/optimum/optimum.csv").iloc[:, 1])

    list_counter_inferences = []
    final_res = []

    # Evaluate the model on each instance and compare against optimal score
    for i in range(len(opts)):
        res2, counter_inferences, _ = evaluate_instance(model, True, args.max_operations, i, dir_path=args.folder)
        print(res2, opts[i], counter_inferences)
        final_res.append((res2 - opts[i]) / opts[i])
        list_counter_inferences.append(counter_inferences)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model on benchmark instances.")
    parser.add_argument('--model_path', type=str, default="model/model.pt", help='Path to the trained model file')
    parser.add_argument('--folder', type=str, default="data/benchmark/taillard", help='Directory path for test data')
    parser.add_argument('--max_operations', type=int, default=10, help='Maximum number of operations per machine')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Number of hidden channels in the model')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the model')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads in the model')

    args = parser.parse_args()
    main(args)
