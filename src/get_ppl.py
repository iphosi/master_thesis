import torch
import tqdm


def modified_perplexity(
    model,
    encodings,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    max_length = model.config.n_positions
    stride = max_length

    nlls = []
    for sample in tqdm(encodings):
        sample_len = sample['input_ids'].size(1)
        sample_nlls = []
        for begin_loc in range(0, sample_len, stride):
            end_loc = min(begin_loc + stride, sample_len)
            # do not create samples with len 1
            if (sample_len - (begin_loc + stride)) == 1:
                #print(sample_len, begin_loc, end_loc)
                end_loc += 1
            input_ids = sample['input_ids'][:, begin_loc:end_loc].to(device)
            if len(input_ids[0]) == 1:
                print('producing nan')
            target_ids = input_ids.clone()
            #target_ids[:,:-1] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids).loss
            sample_nlls.append(outputs)

            if end_loc == sample_len:
                break
        #print(sample_len, len(sample_nlls))
        nlls.append(sum(sample_nlls) / len(sample_nlls))
    return torch.exp(torch.stack(nlls).sum() / len(nlls)).item()
