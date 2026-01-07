import os
import torch
import numpy as np
import gudhi as gd
from transformers import AutoTokenizer, AutoModelForCausalLM
from fava_annot_utils import get_fava_data
from fastchat.model import get_conversation_template
from gudhi.representations import PersistenceImage, BettiCurve, Entropy, Landscape
from pyfzz import pyfzz
import jsonlines

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

from xgboost import XGBClassifier


def read_fava_data(n_samples=460):
    data, _ = get_fava_data(n_samples=n_samples)
    return data

def get_ragtruth_data(n_samples=200):
    train_data = []
    test_data = []
    with jsonlines.open("ragtruth_annotated_data/response.jsonl") as f:
        for row in f:
            if row["model"] == "llama-2-7b-chat" and row["split"] == "train":
                train_data.append(row)
            elif row["model"] == "llama-2-7b-chat" and row["split"] == "test":
                test_data.append(row)
            if len(train_data) >= n_samples:
                break
    return train_data, test_data

def load_model(model_name, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", cache_dir=cache_dir)
    return tokenizer, model

def get_graph_from_attn_mats_in_layer(attn, layer=0, threshold_percentile=90):
    # Average attention across heads
    attn_mats = attn[layer][0]                       # shape: [heads, tokens, tokens]
    avg_attn_mat = attn_mats.mean(dim=0).cpu().numpy()

    # Threshold
    threshold = np.percentile(avg_attn_mat, threshold_percentile)
    mask = avg_attn_mat >= threshold

    # Use only lower-triangular (i > j)
    i, j = np.tril_indices_from(avg_attn_mat, k=-1)
    mask = mask[i, j]

    # Extract edges and weights in one shot
    edge_list = np.stack([i[mask], j[mask]], axis=-1)
    edge_weights = avg_attn_mat[i[mask], j[mask]]

    # Sort by weights
    order = np.argsort(edge_weights)
    edge_list = edge_list[order].tolist()
    edge_weights = edge_weights[order].tolist()

    return edge_list, edge_weights

def intersection_two_lists(list1, list2):
    t1 = map(tuple, list1)
    t2 = map(tuple, list2)
    l = list(map(list, set(t1).intersection(t2)))
    l.sort(key=len)
    return l
        
def set_subtraction_two_lists(list1, list2):
    assert len(list1) >= len(list2)
    t1 = map(tuple, list1)
    t2 = map(tuple, list2)
    l = list(map(list, set(t1).difference(t2)))
    l.sort(key=len)
    return l

def build_zigzag_from_seq_of_graphs(seq_of_graphs, num_vertices):
    zigzag_filt = []
    bar_mapping = {}

    # Use set for curr_simplices for O(1) add/remove
    curr_simplices = set()

    # Add vertices
    for i in range(num_vertices):
        zigzag_filt.append(('i', [i]))
        bar_mapping[len(zigzag_filt) - 1] = 0
        curr_simplices.add((i,))  # store as tuple for consistency

    # Add edges from first graph
    first_edges = set(tuple(edge) for edge in seq_of_graphs[0])
    for edge in first_edges:
        zigzag_filt.append(('i', list(edge)))
        bar_mapping[len(zigzag_filt) - 1] = 0
        curr_simplices.add(edge)

    # Process subsequent graphs
    for idx in range(1, len(seq_of_graphs)):
        prev_edges = set(tuple(edge) for edge in seq_of_graphs[idx - 1])
        curr_edges = set(tuple(edge) for edge in seq_of_graphs[idx])

        # New edges to insert
        new_edges = curr_edges - prev_edges
        for edge in new_edges:
            zigzag_filt.append(('i', list(edge)))
            bar_mapping[len(zigzag_filt) - 1] = idx
            curr_simplices.add(edge)

        # Edges to delete
        deleted_edges = prev_edges - curr_edges
        for edge in deleted_edges:
            zigzag_filt.append(('d', list(edge)))
            bar_mapping[len(zigzag_filt) - 1] = idx
            curr_simplices.discard(edge)  # discard avoids KeyError

    return zigzag_filt, bar_mapping

def parse_bars(bars, bar_mapping, num_graphs):
    h0_bars, h1_bars = [], []
    for bar in bars:
        try: bar_mapping[bar[0]]
        except KeyError: bar_mapping[bar[0]] = num_graphs
        try : bar_mapping[bar[1]]
        except KeyError: bar_mapping[bar[1]] = num_graphs
        if bar_mapping[bar[0]] != bar_mapping[bar[1]]:
            if bar[2] == 0:
                h0_bars.append([bar_mapping[bar[0]], bar_mapping[bar[1]]])
            elif bar[2] == 1:
                h1_bars.append([bar_mapping[bar[0]], bar_mapping[bar[1]]])
    
    return h0_bars, h1_bars

def vectorize_persistence_diagram(persistence, method="persistence_image"):
    if method == "persistence_image":
        pi = PersistenceImage(bandwidth=0.1, resolution=[32, 32])
        vector = pi.fit_transform(persistence)[0]
    elif method == "betti_curve":
        bc = BettiCurve()
        vector = bc.fit_transform(persistence)[0]
    elif method == "entropy":
        en = Entropy()
        vector = en.fit_transform(persistence)[0]
    else:
        raise ValueError("Invalid method. Choose from 'persistence_image', 'betti_curve', or 'entropy'.")
    return vector

def get_bars_from_data(data, model_name, cache_dir, tokenizer, train=True):    
    labels = []
    attn_mats_samples = []
    system_prompt = ""
    h0_bars_all = []
    h1_bars_all = []
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    for i in range(len(data)):
        prompt = data[i]["prompt"]
        response = data[i]["response"]
        label = 1 if data[i]["labels"] else 0
        labels.append(label)

        chat_template = get_conversation_template(model_name)
        chat_template.set_system_message(system_prompt.strip())
        chat_template.messages = []
        chat_template.append_message(chat_template.roles[0], prompt.strip())
        chat_template.append_message(chat_template.roles[1], response.strip())

        full_prompt = chat_template.get_prompt()
        user_prompt = full_prompt.split(response.strip())[0].strip()

        tok_in_u = tokenizer(user_prompt, return_tensors="pt", add_special_tokens=True).input_ids
        tok_in = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=True).input_ids
        
        with torch.no_grad():
            output = model(tok_in, output_attentions=True)
        
        attn = output.attentions
        attn_mats_samples.append(attn)
        num_layers = len(attn)
        num_vertices = attn[0].shape[2]
        print(num_vertices)
        seq_of_graphs = []
        for layer in range(num_layers):
            edge_list, edge_weights = get_graph_from_attn_mats_in_layer(attn, layer)
            seq_of_graphs.append(edge_list)
        
        print('Building zigzag filtration...')
        
        zigzag_filt, bar_mapping = build_zigzag_from_seq_of_graphs(seq_of_graphs, num_vertices)
        fzz = pyfzz()
        print('Computing zigzag persistence...')
        bars = fzz.compute_zigzag(zigzag_filt)
        print('Parsing bars...')
        h0_bars, h1_bars = parse_bars(bars, bar_mapping, num_layers)
        h0_bars = np.array(h0_bars)
        h1_bars = np.array(h1_bars)
        h0_bars_all.append(h0_bars)
        h1_bars_all.append(h1_bars)
        # print(h0_bars)
        # print(h1_bars)
        # break
        if train:
            np.save(f"/scratch/gilbreth/ssamaga/tda_llm_dir/ragtruth/train/h0_bars_{i}.npy", h0_bars)
            np.save(f"/scratch/gilbreth/ssamaga/tda_llm_dir/ragtruth/train/h1_bars_{i}.npy", h1_bars)
        else:
            np.save(f"/scratch/gilbreth/ssamaga/tda_llm_dir/ragtruth/test/h0_bars_{i}.npy", h0_bars)
            np.save(f"/scratch/gilbreth/ssamaga/tda_llm_dir/ragtruth/test/h1_bars_{i}.npy", h1_bars)
    return h0_bars_all, h1_bars_all, labels