import torch
from torchmetrics.functional import pairwise_cosine_similarity
from torchmetrics import PearsonCorrCoef


def get_pearson_scores(
    src_rep_spaces,
    tgt_rep_spaces
):
    pearson = PearsonCorrCoef()
    return list(map(lambda src, tgt: round(pearson(src, tgt).item(), 4), src_rep_spaces, tgt_rep_spaces))


def get_rep_spaces(
    model,
    tokenizer,
    texts,
    num_sample_tokens=2000,
    seed=40
):
    """
    Returns the flattened upper triangular of the similarity matrix in each layer.
    """
    batch_input_ids = [tokenizer(text, return_tensors="pt").input_ids for text in texts]
    batch_hidden_states = [
        model(
            input_ids=input_ids,
            output_hidden_states=True
        ).hidden_states
        for input_ids in batch_input_ids
    ]

    num_layers = len(batch_hidden_states[0])

    rep_spaces = []
    torch.manual_seed(seed)

    for layer in range(num_layers):
        layer_hidden_states = torch.cat(
            [hidden_states[layer][0] for hidden_states in batch_hidden_states],
            dim=0
        )
        num_tokens = layer_hidden_states.size(dim=0)

        if num_tokens >= num_sample_tokens:
            sample_idx = torch.randint(
                low=0,
                high=num_tokens,
                size=(num_sample_tokens,)
            )
            sim_matrix = pairwise_cosine_similarity(layer_hidden_states[sample_idx, :])

        else:
            sim_matrix = pairwise_cosine_similarity(layer_hidden_states)

        rep_space = torch.triu(sim_matrix).view(-1)
        rep_spaces.append(rep_space)

    return rep_spaces
