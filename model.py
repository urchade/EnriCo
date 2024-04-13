import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from base import InstructBase
from modules.evaluator import Evaluator
from modules.layers import FilteringLayer, RelationRep, RefineLayer, ScorerLayer, LstmSeq2SeqEncoder
from modules.span_rep import SpanRepLayer
from modules.token_rep import TokenRepLayer
from modules.utils import get_ground_truth_relations, get_relations, get_relation_with_span, _get_candidates


class EnriCo(InstructBase, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        # [ENT] token
        self.rel_token = "<<REL>>"
        self.sep_token = "<<SEP>>"

        # usually a pretrained bidirectional transformer, returns first subtoken representation
        self.token_rep_layer = TokenRepLayer(model_name=config.model_name, fine_tune=config.fine_tune,
                                             subtoken_pooling=config.subtoken_pooling, hidden_size=config.hidden_size,
                                             add_tokens=[self.rel_token, self.sep_token])

        # hierarchical representation of tokens (Zaratiana et al, 2022)
        # https://arxiv.org/pdf/2203.14710.pdf
        self.rnn = LstmSeq2SeqEncoder(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=1,
            bidirectional=True
        )

        # span representation
        # we have a paper to study span representation for ner
        # Zaratiana et al, 2022: https://aclanthology.org/2022.umios-1.1/
        self.span_rep_layer = SpanRepLayer(
            span_mode=config.span_mode,
            hidden_size=config.hidden_size,
            max_width=config.max_width,
            dropout=config.dropout
        )

        # prompt representation (FFN)
        self.prompt_rep_layer = nn.Linear(config.hidden_size, config.hidden_size)

        # filtering layer for spans and relations
        self._span_filtering = FilteringLayer(config.hidden_size)
        self._rel_filtering = FilteringLayer(config.hidden_size)

        # relation representation
        self.relation_rep = RelationRep(config.hidden_size, config.dropout, config.ffn_mul)

        # refine span representation
        if self.config.refine_span:
            self.refine_span = RefineLayer(
                config.hidden_size, config.hidden_size // 64, num_layers=1, ffn_mul=config.ffn_mul,
                dropout=config.dropout,
                read_only=True
            )

        # refine relation representation
        if self.config.refine_relation:
            self.refine_relation = RefineLayer(
                config.hidden_size, config.hidden_size // 64, num_layers=1, ffn_mul=config.ffn_mul,
                dropout=config.dropout,
                read_only=True
            )

        # refine prompt representation
        if self.config.refine_prompt:
            self.refine_prompt = RefineLayer(
                config.hidden_size, config.hidden_size // 64, num_layers=2, ffn_mul=config.ffn_mul,
                dropout=config.dropout,
                read_only=True
            )

        # scoring layer
        self.scorer = ScorerLayer(scoring_type=config.scorer, hidden_size=config.hidden_size, dropout=config.dropout)

    def get_optimizer(self, lr_encoder, lr_others, freeze_token_rep=False):
        """
        Parameters:
        - lr_encoder: Learning rate for the encoder layer.
        - lr_others: Learning rate for all other layers.
        - freeze_token_rep: whether the token representation layer should be frozen.
        """
        param_groups = [
            # encoder
            {"params": self.rnn.parameters(), "lr": lr_others},
            # projection layers
            {"params": self.span_rep_layer.parameters(), "lr": lr_others},
            # prompt representation
            {"params": self.prompt_rep_layer.parameters(), "lr": lr_others},
            # filtering layers
            {"params": self._span_filtering.parameters(), "lr": lr_others},
            {"params": self._rel_filtering.parameters(), "lr": lr_others},
            {'params': self.relation_rep.parameters(), 'lr': lr_others},
            # scorer
            {'params': self.scorer.parameters(), 'lr': lr_others}
        ]

        if self.config.refine_span:
            # If refine_span layer should not be frozen, add it to the optimizer with its learning rate
            param_groups.append({"params": self.refine_span.parameters(), "lr": lr_others})

        if self.config.refine_relation:
            # If refine_relation layer should not be frozen, add it to the optimizer with its learning rate
            param_groups.append({"params": self.refine_relation.parameters(), "lr": lr_others})

        if self.config.refine_prompt:
            # If refine_prompt layer should not be frozen, add it to the optimizer with its learning rate
            param_groups.append({"params": self.refine_prompt.parameters(), "lr": lr_others})

        if not freeze_token_rep:
            # If token_rep_layer should not be frozen, add it to the optimizer with its learning rate
            param_groups.append({"params": self.token_rep_layer.parameters(), "lr": lr_encoder})
        else:
            # If token_rep_layer should be frozen, explicitly set requires_grad to False for its parameters
            for param in self.token_rep_layer.parameters():
                param.requires_grad = False

        optimizer = torch.optim.AdamW(param_groups)

        return optimizer

    def compute_score_train(self, x):
        span_idx = x['span_idx'] * x['span_mask'].unsqueeze(-1)

        new_length = x['seq_length'].clone()
        new_tokens = []
        all_len_prompt = []
        num_classes_all = []

        # add prompt to the tokens
        for i in range(len(x['tokens'])):
            all_types_i = list(x['classes_to_id'][i].keys())
            # multiple relation types in all_types. Prompt is appended at the start of tokens
            relation_prompt = []
            num_classes_all.append(len(all_types_i))
            # add relation types to prompt
            for relation_type in all_types_i:
                relation_prompt.append(self.rel_token)  # [ENT] token
                relation_prompt.append(relation_type)  # relation type
            relation_prompt.append(self.sep_token)  # [SEP] token

            # prompt format:
            # [ENT] relation_type [ENT] relation_type ... [ENT] relation_type [SEP]

            # add prompt to the tokens
            tokens_p = relation_prompt + x['tokens'][i]

            # input format:
            # [ENT] relation_type_1 [ENT] relation_type_2 ... [ENT] relation_type_m [SEP] token_1 token_2 ... token_n

            # update length of the sequence (add prompt length to the original length)
            new_length[i] = new_length[i] + len(relation_prompt)
            # update tokens
            new_tokens.append(tokens_p)
            # store prompt length
            all_len_prompt.append(len(relation_prompt))

        # create a mask using num_classes_all (0, if it exceeds the number of classes, 1 otherwise)
        max_num_classes = max(num_classes_all)
        relation_type_mask = torch.arange(max_num_classes).unsqueeze(0).expand(len(num_classes_all), -1).to(
            x['span_mask'].device)
        relation_type_mask = relation_type_mask < torch.tensor(num_classes_all).unsqueeze(-1).to(
            x['span_mask'].device)  # [batch_size, max_num_classes]

        # compute all token representations
        bert_output = self.token_rep_layer(new_tokens, new_length)
        word_rep_w_prompt = bert_output["embeddings"]  # embeddings for all tokens (with prompt)
        mask_w_prompt = bert_output["mask"]  # mask for all tokens (with prompt)

        # get word representation (after [SEP]), mask (after [SEP]) and relation type representation (before [SEP])
        word_rep = []  # word representation (after [SEP])
        mask = []  # mask (after [SEP])
        rel_type_rep = []  # relation type representation (before [SEP])
        for i in range(len(x['tokens'])):
            prompt_relation_length = all_len_prompt[i]  # length of prompt for this example
            # get word representation (after [SEP])
            word_rep.append(word_rep_w_prompt[i, prompt_relation_length:prompt_relation_length + x['seq_length'][i]])
            # get mask (after [SEP])
            mask.append(mask_w_prompt[i, prompt_relation_length:prompt_relation_length + x['seq_length'][i]])

            # get relation type representation (before [SEP])
            rel_rep = word_rep_w_prompt[i, :prompt_relation_length - 1]  # remove [SEP]
            rel_rep = rel_rep[0::2]  # it means that we take every second element starting from the second one
            rel_type_rep.append(rel_rep)

        # padding for word_rep, mask and rel_type_rep
        word_rep = pad_sequence(word_rep, batch_first=True)  # [batch_size, seq_len, hidden_size]
        mask = pad_sequence(mask, batch_first=True)  # [batch_size, seq_len]
        rel_type_rep = pad_sequence(rel_type_rep, batch_first=True)  # [batch_size, len_types, hidden_size]

        # compute span representation
        word_rep = self.rnn(word_rep, mask)
        span_rep = self.span_rep_layer(word_rep, span_idx)

        # compute final relation type representation (FFN)
        rel_type_rep = self.prompt_rep_layer(rel_type_rep)  # (batch_size, len_types, hidden_size)
        num_classes = rel_type_rep.shape[1]  # number of relation types

        return span_rep, num_classes, rel_type_rep, relation_type_mask, (word_rep, mask)

    def forward(self, x, prediction_mode=False):

        # clone span_label
        span_label = x['span_label'].clone()

        # compute span representation
        if prediction_mode:
            device = next(self.parameters()).device
            span_rep, num_classes, rel_type_rep, relation_type_mask, (word_rep, word_mask) = self.compute_score_eval(x,
                                                                                                                     device)
            # set relation_type_mask to tensor of ones
            relation_type_mask = torch.ones(rel_type_rep.shape[0], num_classes).to(device)
        else:
            span_rep, num_classes, rel_type_rep, relation_type_mask, (word_rep, word_mask) = self.compute_score_train(x)

        B, L, K, D = span_rep.shape
        span_rep = span_rep.view(B, L * K, D)

        # filtering scores for spans
        filter_score_span, filter_loss_span = self._span_filtering(span_rep, x['span_label'])

        # number of candidates
        max_top_k = L + self.config.add_top_k

        if L > self.config.max_top_k:
            max_top_k = self.config.max_top_k + self.config.add_top_k

        # filtering scores for spans
        _, sorted_idx = torch.sort(filter_score_span, dim=-1, descending=True)

        # Get candidate spans and labels
        candidate_span_rep, candidate_span_label, candidate_span_mask, candidate_spans_idx = [
            _get_candidates(sorted_idx, el, topk=max_top_k)[0] for el in
            [span_rep, span_label, x['span_mask'], x['span_idx']]]

        # configure masks for entity #############################################
        ##########################################################################
        top_k_lengths = x["seq_length"].clone() + self.config.add_top_k
        arange_topk = torch.arange(max_top_k, device=span_rep.device)
        masked_fill_cond = arange_topk.unsqueeze(0) >= top_k_lengths.unsqueeze(-1)
        candidate_span_mask.masked_fill_(masked_fill_cond, 0)
        candidate_span_label.masked_fill_(masked_fill_cond, -1)
        ##########################################################################
        ##########################################################################

        if self.config.refine_span:
            # refine span representation
            candidate_span_rep = self.refine_span(
                candidate_span_rep, word_rep, candidate_span_mask, word_mask
            )

        # get ground truth relations
        relation_classes = get_ground_truth_relations(x, candidate_spans_idx, candidate_span_label)

        # representation of relations
        rel_rep = self.relation_rep(candidate_span_rep)  # [B, topk, topk, D]

        # filtering scores for relations
        filter_score_rel, filter_loss_rel = self._rel_filtering(
            rel_rep.view(B, max_top_k * max_top_k, -1), relation_classes)

        # filtering scores for relation pairs
        _, sorted_idx_pair = torch.sort(filter_score_rel, dim=-1, descending=True)

        # Get candidate pairs and labels
        candidate_pair_rep, candidate_pair_label = [_get_candidates(sorted_idx_pair, el, topk=max_top_k)[0] for el
                                                    in
                                                    [rel_rep.view(B, max_top_k * max_top_k, -1),
                                                     relation_classes.view(B, max_top_k * max_top_k)]]

        topK_rel_idx = sorted_idx_pair[:, :max_top_k]

        #######################################################
        candidate_pair_label.masked_fill_(masked_fill_cond, -1)
        #######################################################

        # refine relation representation ##############################################
        candidate_pair_mask = candidate_pair_label > -1
        ################################################################################

        if self.config.refine_relation:
            # refine relation representation
            candidate_pair_rep = self.refine_relation(
                candidate_pair_rep, word_rep, candidate_pair_mask, word_mask
            )

        # refine relation representation with relation type representation ############
        rel_type_rep = self.refine_prompt(
            rel_type_rep, candidate_pair_rep, relation_type_mask, candidate_pair_mask
        )
        ################################################################################

        # compute scores
        scores = self.scorer(candidate_pair_rep, rel_type_rep)  # [B, N, C]

        if prediction_mode:
            return {"relation_logits": scores, "candidate_spans_idx": candidate_spans_idx,
                    "candidate_pair_label": candidate_pair_label,
                    "max_top_k": max_top_k, "topK_rel_idx": topK_rel_idx}

        # loss for filtering classifier
        logits_label = scores.view(-1, num_classes)
        labels = candidate_pair_label.view(-1)  # (batch_size * num_spans)
        mask_label = labels != -1  # (batch_size * num_spans)
        labels.masked_fill_(~mask_label, 0)  # Set the labels of padding tokens to 0

        # one-hot encoding
        labels_one_hot = torch.zeros(labels.size(0), num_classes + 1, dtype=torch.float32).to(scores.device)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)  # Set the corresponding index to 1
        labels_one_hot = labels_one_hot[:, 1:]  # Remove the first column

        # loss for relation classifier
        relation_loss = F.binary_cross_entropy_with_logits(logits_label, labels_one_hot,
                                                           reduction='none')
        # mask loss using relation_type_mask (B, C)
        masked_loss = relation_loss.view(B, -1, num_classes) * relation_type_mask.unsqueeze(1)
        relation_loss = masked_loss.view(-1, num_classes)
        # expand mask_label to relation_loss
        mask_label = mask_label.unsqueeze(-1).expand_as(relation_loss)
        # put lower loss for in label_one_hot (2 for positive, 1 for negative)
        weight_c = labels_one_hot + 1
        # apply mask
        relation_loss = relation_loss * mask_label.float() * weight_c
        relation_loss = relation_loss.sum()

        return filter_loss_span + filter_loss_rel + relation_loss

    @torch.no_grad()
    def compute_score_eval(self, x, device):
        # check if classes_to_id is dict
        assert isinstance(x['classes_to_id'], dict), "classes_to_id must be a dict"
        assert isinstance(x['id_to_classes'], dict), "id_to_classes must be a dict"

        span_idx = (x['span_idx'] * x['span_mask'].unsqueeze(-1)).to(device)

        all_types = list(x['classes_to_id'].keys())
        # multiple relation types in all_types. Prompt is appended at the start of tokens
        relation_prompt = []

        # add relation types to prompt
        for relation_type in all_types:
            relation_prompt.append(self.rel_token)
            relation_prompt.append(relation_type)

        relation_prompt.append(self.sep_token)

        prompt_relation_length = len(relation_prompt)

        # add prompt
        tokens_p = [relation_prompt + tokens for tokens in x['tokens']]
        seq_length_p = x['seq_length'] + prompt_relation_length

        out = self.token_rep_layer(tokens_p, seq_length_p)

        word_rep_w_prompt = out["embeddings"]
        mask_w_prompt = out["mask"]

        # remove prompt
        word_rep = word_rep_w_prompt[:, prompt_relation_length:, :]
        mask = mask_w_prompt[:, prompt_relation_length:]

        # get_relation_type_rep
        relation_type_rep = word_rep_w_prompt[:, :prompt_relation_length - 1, :]
        # extract [REL] tokens (which are at even positions in relation_type_rep)
        relation_type_rep = relation_type_rep[:, 0::2, :]

        relation_type_rep = self.prompt_rep_layer(relation_type_rep)  # (batch_size, len_types, hidden_size)

        word_rep = self.rnn(word_rep, mask)

        span_rep = self.span_rep_layer(word_rep, span_idx)

        num_classes = relation_type_rep.shape[1]

        return span_rep, num_classes, relation_type_rep, None, (word_rep, mask)

    def predict(self, x, threshold=0.5, output_confidence=False):
        out = self.forward(x, prediction_mode=True)
        relations = get_relations(x, out["relation_logits"], out["topK_rel_idx"], out["max_top_k"],
                                  out["candidate_spans_idx"], threshold=threshold, output_confidence=output_confidence)
        return relations

    def evaluate(self, test_data, threshold=0.5, batch_size=12, relation_types=None):
        self.eval()
        data_loader = self.create_dataloader(test_data, batch_size=batch_size, relation_types=relation_types,
                                             shuffle=False)
        device = next(self.parameters()).device
        all_preds = []
        all_trues = []
        for x in data_loader:
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(device)
            batch_predictions = self.predict(x, threshold)
            all_preds.extend(batch_predictions)
            all_trues.extend(get_relation_with_span(x))
        evaluator = Evaluator(all_trues, all_preds)
        out, f1 = evaluator.evaluate()
        return out, f1

    def save_pretrained(
            self,
            save_directory: Union[str, Path],
            *,
            config: Optional[Union[dict, "DataclassInstance"]] = None,
            repo_id: Optional[str] = None,
            push_to_hub: bool = False,
            **push_to_hub_kwargs,
    ) -> Optional[str]:
        """
        Save weights in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            config (`dict` or `DataclassInstance`, *optional*):
                Model configuration specified as a key/value dictionary or a dataclass instance.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            kwargs:
                Additional key word arguments passed along to the [`~ModelHubMixin.push_to_hub`] method.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # save model weights/files
        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")

        # save config (if provided)
        if config is None:
            config = self.config
        if config is not None:
            if isinstance(config, argparse.Namespace):
                config = vars(config)
            (save_directory / "config.json").write_text(json.dumps(config, indent=2))

        # push to the Hub if required
        if push_to_hub:
            kwargs = push_to_hub_kwargs.copy()  # soft-copy to avoid mutating input
            if config is not None:  # kwarg for `push_to_hub`
                kwargs["config"] = config
            if repo_id is None:
                repo_id = save_directory.name  # Defaults to `save_directory` name
            return self.push_to_hub(repo_id=repo_id, **kwargs)
        return None

    @classmethod
    def _from_pretrained(
            cls,
            *,
            model_id: str,
            revision: Optional[str],
            cache_dir: Optional[Union[str, Path]],
            force_download: bool,
            proxies: Optional[Dict],
            resume_download: bool,
            local_files_only: bool,
            token: Union[str, bool, None],
            map_location: str = "cpu",
            strict: bool = False,
            **model_kwargs,
    ):

        # 2. Newer format: Use "pytorch_model.bin" and "gliner_config.json"
        model_file = Path(model_id) / "pytorch_model.bin"
        if not model_file.exists():
            model_file = hf_hub_download(
                repo_id=model_id,
                filename="pytorch_model.bin",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        config_file = Path(model_id) / "config.json"
        if not config_file.exists():
            config_file = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        config = load_config_as_namespace(config_file)
        model = cls(config)
        state_dict = torch.load(model_file, map_location=torch.device(map_location))
        model.load_state_dict(state_dict, strict=strict, assign=True)
        model.to(map_location)
        return model

    def to(self, device):
        super().to(device)
        import flair
        flair.device = device
        return self


def load_config_as_namespace(config_file):
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)
