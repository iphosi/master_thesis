import torch
from transformers import AdapterTrainer
from loss_fct import contrastive_loss


class ContrastiveTrainer(AdapterTrainer):
    def __init__(
        self,
        margin=0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
    ):
        input_ids = inputs.get("input_ids")
        bsz, seqlen = input_ids.size()

        labels = inputs.get('labels')
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.get("logits")
        assert logits.size() == torch.Size([bsz, seqlen, model.config.vocab_size])

        if self.label_smoother:
            mle_loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            mle_loss = outputs.get("loss")

        # Contrastive Loss
        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.size() == torch.Size([bsz, seqlen, model.config.hidden_size])

        norm_rep = last_hidden_states / last_hidden_states.norm(dim=2, keepdim=True)
        cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1, 2))
        assert cosine_scores.size() == torch.Size([bsz, seqlen, seqlen])

        cl_loss = contrastive_loss(
            self.margin,
            cosine_scores,
            input_ids,
            model.config.pad_token_id,
            prefix_len=0
        )

        loss = mle_loss + cl_loss

        return (loss, outputs) if return_outputs else loss