from configuration_phi import PhiConfig
from modeling_phi import PhiForCausalLM
import numpy as np
import argparse
import os
import torch
from transformers import AutoModelForCausalLM


def uniform_selection(weights, old_dim, new_dim, vocab_size):
    new_weights = {}
    for key in weights:
        original = np.array(weights[key])
        new_dim_shapes = [size if size == vocab_size else int(size * new_dim / old_dim) for size in original.shape]
        indices = [np.linspace(0, o-1, n, dtype=int) for o, n in zip(original.shape, new_dim_shapes)]
        new_weights[key] = torch.Tensor(original[tuple(np.ix_(*indices))])
    return new_weights


def initialize_model(model, smol):
    original = model.state_dict()
    smol_dict = smol.state_dict()

    for key in original.keys():
        if key in smol_dict.keys():
            smol_dict[key] = original[key]

    if model.config.n_embd != smol.config.n_embd:
        print("Model dimensions do not match. We need to perform weight selection")
        smol_dict = uniform_selection(smol_dict, model.config.n_embd, smol.config.n_embd, smol.config.vocab_size)

    smol.load_state_dict(smol_dict)
    return smol

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initializing small models with larger ones")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2",
        help="The model to use for initialization",
    )
    parser.add_argument(
        "--n-layer",
        type=int,
        default=None,
        help="The number of layers to use in the smaller model",
    )
    parser.add_argument(
        "--n-embd",
        type=int,
        default=None,
        help="The dimension of the smaller model",
    )
    parser.add_argument(
        "--n-head",
        type=int,
        default=None,
        help="The number of heads to use in the smaller model",
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
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True, revision="834565c23f9b28b96ccbeabe614dd906b6db551a")
    config = PhiConfig.from_pretrained("microsoft/phi-2", trust_remote_code=True, revision="834565c23f9b28b96ccbeabe614dd906b6db551a")
    config.n_layer = args.n_layers if args.n_layers is not None else config.n_layer
    config.n_embd = args.n_embd if args.n_embd is not None else config.n_embd
    config.n_head = args.n_heads if args.n_heads is not None else config.n_head
    smol = PhiForCausalLM(config)
    nparams = sum(
        x.numel() for x in smol.parameters() if x.requires_grad
    )
    print("Initializing a transformer with {:,} parameters".format(nparams))
    # input y to continue, n to cancel, or o to use random weights
    answer = input("Continue? [y/n/o]")
    if answer == "n":
        exit()
    elif answer == "o":
        print("Using random weights")
    else:
        smol = initialize_model(model, smol)

    # check if the output path exists
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # save the model
    smol.save_pretrained(args.output_path)
    smol.config.save_pretrained(args.output_path)