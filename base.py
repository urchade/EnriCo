import random
from collections import defaultdict

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class InstructBase(nn.Module):
    """
    Base class for preprocessing and dataloader
    """

    def __init__(self, base_config):
        super().__init__()

        # classes: label name [eg. PERS, ORG, ...]
        self.max_width = base_config.max_width
        self.base_config = base_config

    def get_dict(self, spans):
        dict_tag = defaultdict(int)
        for span in spans:
            dict_tag[(span[0], span[1])] = 1
        return dict_tag

    def preprocess_spans(self, tokens, ner, rel):

        max_len = self.base_config.max_len

        if len(tokens) > max_len:
            length = max_len
            tokens = tokens[:max_len]
        else:
            length = len(tokens)

        spans_idx = []
        for i in range(length):
            spans_idx.extend([(i, i + j) for j in range(self.max_width)])

        dict_lab = self.get_dict(ner) if ner else defaultdict(int)

        # 0 for null labels
        span_label = torch.LongTensor([dict_lab[i] for i in spans_idx])
        spans_idx = torch.LongTensor(spans_idx)

        # mask for valid spans
        valid_span_mask = spans_idx[:, 1] > length - 1

        # mask invalid positions
        span_label = span_label.masked_fill(valid_span_mask, -1)

        return {
            'tokens': tokens,
            'span_idx': spans_idx,
            'span_label': span_label,
            'seq_length': length,
            'entities': ner,
            'relations': rel,
        }

    def collate_fn(self, batch_list, relation_types=None):

        # batch = [self.preprocess_spans(['tokenized_text'], b['spans'], b['relations'])
        # for b in batch_list]

        if relation_types is None:

            negs = self.get_negatives(batch_list, 100)

            rel_to_id = []
            id_to_rel = []

            for b in batch_list:
                random.shuffle(negs)

                # negs = negs[:sampled_neg]
                max_neg_type_ratio = int(self.base_config.max_neg_type_ratio)

                if max_neg_type_ratio == 0:
                    # no negatives
                    neg_type_ratio = 0
                else:
                    neg_type_ratio = random.randint(0, max_neg_type_ratio)

                if neg_type_ratio == 0:
                    # no negatives
                    negs_i = []
                else:
                    negs_i = negs[:len(b['relations']) * neg_type_ratio]

                # this is the list of all possible entity types (positive and negative)
                types = list(set([el[-1] for el in b['relations']] + negs_i))

                # shuffle (every epoch)
                random.shuffle(types)

                if len(types) != 0:
                    # prob of higher number shoul
                    # random drop
                    if self.base_config.random_drop:
                        num_ents = random.randint(1, len(types))
                        types = types[:num_ents]

                # maximum number of entities types
                types = types[:int(self.base_config.max_types)]

                # supervised training
                if "label" in b:
                    types = sorted(b["label"])

                class_to_id = {k: v for v, k in enumerate(types, start=1)}
                id_to_class = {k: v for v, k in class_to_id.items()}
                rel_to_id.append(class_to_id)
                id_to_rel.append(id_to_class)

            batch = [
                self.preprocess_spans(b["tokenized_text"], b["spans"], b["relations"]) for i, b in enumerate(batch_list)
            ]

        else:
            rel_to_id = {k: v for v, k in enumerate(relation_types, start=1)}
            id_to_rel = {k: v for v, k in rel_to_id.items()}
            batch = [
                self.preprocess_spans(b["tokenized_text"], b["spans"], b["relations"]) for b in batch_list
            ]

        span_idx = pad_sequence(
            [b['span_idx'] for b in batch], batch_first=True, padding_value=0
        )

        span_label = pad_sequence(
            [el['span_label'] for el in batch], batch_first=True, padding_value=-1
        )

        return {
            'seq_length': torch.LongTensor([el['seq_length'] for el in batch]),
            'span_idx': span_idx,
            'tokens': [el['tokens'] for el in batch],
            'span_mask': span_label != -1,
            'span_label': span_label,
            'entities': [el['entities'] for el in batch],
            'relations': [el['relations'] for el in batch],
            'classes_to_id': rel_to_id,
            'id_to_classes': id_to_rel,
        }

    @staticmethod
    def get_negatives(batch_list, sampled_neg=50):
        rel_types = []
        for b in batch_list:
            types = set([el[-1] for el in b['relations']])
            rel_types.extend(list(types))
        ent_types = list(set(rel_types))
        # sample negatives
        random.shuffle(ent_types)
        return ent_types[:sampled_neg]

    def create_dataloader(self, data, relation_types=None, **kwargs):
        return DataLoader(data, collate_fn=lambda x: self.collate_fn(x, relation_types), **kwargs)


if __name__ == '__main__':
    path = "/Users/urchadezaratiana/Documents/remote-server/GLiREL/data/nuner_relation.json"

    # Open data
    import json

    with open(path, "rb") as f:
        dataset = json.load(f)

    from model import GLiRel

    # import simple namespace
    from types import SimpleNamespace

    base_config = SimpleNamespace(model_name="bert-base-uncased",
                                  max_len=100, max_types=20,
                                  max_neg_type_ratio=2,
                                  random_drop=True,
                                  max_width=5,
                                  fine_tune=False,
                                  subtoken_pooling="first",
                                  hidden_size=768,
                                  dropout=0.1,
                                  span_mode="markerV0")

    model = GLiRel(base_config)

    loader = model.create_dataloader(dataset, batch_size=5, shuffle=True, num_workers=0)

    x = next(iter(loader))

    model.eval()
    out = model.forward(x, prediction_mode=True)

    print(out)

    # from modules.run_evaluation import get_for_one_path
    #
    # out = get_for_one_path("/Users/urchadezaratiana/Documents/remote-server/ie_data/RE/SciERC", model)
    #
    # print(out)
