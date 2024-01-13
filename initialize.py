import numpy as np
import mlx
from mlx.utils import tree_flatten
import argparse
import model as phi2
import os

def get_weights(model, num_layers = None):
    parameters = {}
    for key, value in tree_flatten(model.parameters()):
        split_key = key.split(".")
        num = 0
        for i, k in enumerate(split_key):
            if k == "h":
                num = int(split_key[i+1])
                break
        if num_layers and num >= num_layers:
            continue
        parameters[key] = value
    return parameters

def uniform_selection(weights, old_dim, new_dim, vocab_size):
    new_weights = {}
    for key in weights:
        original = np.array(weights[key])
        new_dim_shapes = [size if size == vocab_size else int(size * new_dim / old_dim) for size in original.shape]
        indices = [np.linspace(0, o-1, n, dtype=int) for o, n in zip(original.shape, new_dim_shapes)]
        new_weights[key] = mlx.core.array(original[tuple(np.ix_(*indices))])
    return new_weights

def copy_weights(model, smaller_model):
    weights = get_weights(model, smaller_model.config.num_layers)
    # we need to check if the dimensions match
    if model.config.model_dim != smaller_model.config.model_dim:
        print("Model dimensions do not match. We need to perform weight selection")
        weights = uniform_selection(weights, model.config.model_dim, smaller_model.config.model_dim, smaller_model.config.num_vocab)
    # we need to adjust the weights by randomly s
    smaller_model.load_weights(list(weights.items()))
    return smaller_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initializing small models with larger ones")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2",
        help="The model to use for initialization",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="The number of layers to use in the smaller model",
    )
    parser.add_argument(
        "--model-dim",
        type=int,
        default=2560,
        help="The dimension of the smaller model",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="initialized_model",
        help="The path to the output model",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="model",
        help="The name of the output model",
    )

    args = parser.parse_args()
    print(f"Initializing a transformer from {args.model}")
    model, tokenizer = phi2.load(args.model)
    smol_model = phi2.Model(phi2.ModelArgs(num_layers=args.num_layers, model_dim=args.model_dim))
    nparams = sum(
        x.size for k, x in tree_flatten(smol_model.parameters())
    )
    print("Initializing a transformer with {:,} parameters".format(nparams))
    smaller_model = copy_weights(model, smol_model)

    # check if the output path exists
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # save the model
    mlx.core.save_safetensors(f"{args.output_path}/{args.output_file}", get_weights(smol_model))

    





