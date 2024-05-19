import argparse
from train import start_train
from generate_expert_obs import generate_expert_obs, read_files
import pickle

def main(args):
    dir_path = args.dir_path
    max_operations = args.max_operations

    try:
        with open("expert_observations.pkl", "rb") as file:
            expert_observations = pickle.load(file)
    except FileNotFoundError:
        name_files = read_files(dir_path=dir_path)
        expert_observations = generate_expert_obs(name_files, max_operations)
        with open("expert_observations.pkl", "wb") as file:
            pickle.dump(expert_observations, file)

    start_train(
        expert_observations,
        max_operations,
        args.epochs,
        args.learning_rate,
        args.batch_size,
        args.embedding_size,
        args.heads,
        args.layers
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training with specified parameters.")
    parser.add_argument('--dir_path', type=str, default="data/train", help='Directory path for training data')
    parser.add_argument('--max_operations', type=int, default=10, help='Maximum number of operations in the env')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size for the model')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--layers', type=int, default=4, help='Layers of the GNN')

    args = parser.parse_args()
    main(args)
