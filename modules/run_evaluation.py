import glob
import json
import os
import random
import re

import torch
from tqdm import tqdm


def open_content(path):
    paths = glob.glob(os.path.join(path, "*.json"))
    train, dev, test, labels = None, None, None, None
    for p in paths:
        if "train" in p:
            with open(p, "r") as f:
                train = json.load(f)
        elif "dev" in p:
            with open(p, "r") as f:
                dev = json.load(f)
        elif "test" in p:
            with open(p, "r") as f:
                test = json.load(f)
        elif "labels" in p:
            with open(p, "r") as f:
                labels = json.load(f)
    return train, dev, test, labels  # [LABEL_MAP[i] for i in ]


def tokenize(text):
    tokens = []

    for match in re.finditer(r'\w+(?:[-_]\w+)*|\S', text):
        tokens.append(match.group())

    return tokens


def process_data(data):
    # Assuming tokenize is defined elsewhere
    rel_data = []  # Initialize an empty list to hold the processed relation data

    for el in data:
        tokens = tokenize(el["sentence"])
        all_relations = []
        all_spans = []

        try:
            for rel in el["relations"]:
                head = tokenize(rel["head"]["name"])  # Tokenize the head entity
                tail = tokenize(rel["tail"]["name"])  # Tokenize the tail entity
                r_type = rel["type"]  # LABEL_MAP[rel["type"]]  # Type of relation

                # Initialize lists to hold all head and tail positions
                all_heads = []
                all_tails = []

                # Find positions of head and tail in tokens
                for i in range(len(tokens)):
                    if tokens[i] == head[0] and tokens[i:i + len(head)] == head:
                        head_pos = (i, i + len(head) - 1)
                        all_heads.append(head_pos)

                for i in range(len(tokens)):
                    if tokens[i] == tail[0] and tokens[i:i + len(tail)] == tail:
                        tail_pos = (i, i + len(tail) - 1)
                        all_tails.append(tail_pos)

                # Create all possible relations
                for head_pos in all_heads:
                    for tail_pos in all_tails:
                        all_relations.append([r_type, (head_pos, tail_pos)])

                all_spans.extend(all_heads)
                all_spans.extend(all_tails)
        except:
            continue

        # Format relations
        relations = []
        for rel in all_relations:
            head_pos = all_spans.index(rel[1][0])
            tail_pos = all_spans.index(rel[1][1])
            relations.append((head_pos, tail_pos, rel[0]))

        # Append processed data to rel_data
        rel_data.append({
            "tokenized_text": tokens,
            "spans": all_spans,
            "relations": relations
        })

    return rel_data


# create dataset
def create_dataset(path):
    train, dev, test, labels = open_content(path)
    try:
        train_dataset = process_data(train)
        dev_dataset = process_data(dev)
    except:
        train_dataset = []
        dev_dataset = []
    test_dataset = process_data(test)
    return train_dataset, dev_dataset, test_dataset, labels


@torch.no_grad()
def get_for_one_path(path, model):
    # load the dataset
    _, _, test_dataset, relation_types = create_dataset(path)

    data_name = path.split("/")[-1]  # get the name of the dataset

    # evaluate the model
    results, f1 = model.evaluate(test_dataset[:10], threshold=0.5, batch_size=12,
                                 relation_types=relation_types)
    return data_name, results, f1


def get_for_all_path(model, steps, log_dir, data_paths):
    all_paths = glob.glob(f"{data_paths}/*")

    all_paths = sorted(all_paths)

    # move the model to the device
    device = next(model.parameters()).device
    model.to(device)
    # set the model to eval mode
    model.eval()

    # log the results
    save_path = os.path.join(log_dir, "results.txt")

    with open(save_path, "a") as f:
        f.write("##############################################\n")
        # write step
        f.write("step: " + str(steps) + "\n")

    all_results = {}  # without crossNER

    for p in tqdm(all_paths):
        if "sample_" not in p:
            try:
                data_name, results, f1 = get_for_one_path(p, model)
            except:
                continue
            # write to file
            with open(save_path, "a") as f:
                f.write(data_name + "\n")
                f.write(str(results) + "\n")

            all_results[data_name] = f1

    avg_all = sum(all_results.values()) / len(all_results)

    save_path_table = os.path.join(log_dir, "tables.txt")

    # results for all datasets except crossNER
    table_bench_all = ""
    for k, v in all_results.items():
        table_bench_all += f"{k:20}: {v:.1%}\n"
    # (20 size aswell for average i.e. :20)
    table_bench_all += f"{'Average':20}: {avg_all:.1%}"

    # write to file
    with open(save_path_table, "a") as f:
        f.write("##############################################\n")
        f.write("step: " + str(steps) + "\n")
        f.write("Table for all datasets\n")
        f.write(table_bench_all + "\n\n")
        f.write("##############################################\n\n")


def sample_train_data(data_paths, sample_size=10000):
    all_paths = glob.glob(f"{data_paths}/*")

    all_paths = sorted(all_paths)

    new_train = []
    # take 10k samples from each dataset
    for p in tqdm(all_paths):
        train, dev, test, labels = create_dataset(p)

        # add label key to the train data
        for i in range(len(train)):
            train[i]["label"] = labels

        random.shuffle(train)
        train = train[:sample_size]
        new_train.extend(train)

    return new_train
