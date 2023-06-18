import os
import re
import json
import random
import logging
import math

from tqdm import tqdm
from copy import deepcopy

logger = logging.getLogger(__name__)

RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def remove_articles(_text):
    return RE_ART.sub(' ', _text)


def white_space_fix(_text):
    return ' '.join(_text.split())


def remove_punc(_text):
    return RE_PUNC.sub(' ', _text)  # convert punctuation to spaces


def lower(_text):
    return _text.lower()


def normalize(text):
    """Lower text and remove punctuation, articles and extra whitespace. """
    return white_space_fix(remove_articles(remove_punc(lower(text))))


def pad_ids(arrays, padding, max_length=-1):
    if max_length < 0:
        max_length = max(list(map(len, arrays)))
    
    arrays = [
        array + [padding] * (max_length - len(array))
        for array in arrays
    ]

    return arrays


def truncate_sequences(sequences, max_length):
    words_to_cut = sum(list(map(len, sequences))) - max_length
    if words_to_cut <= 0:
        return sequences

    while words_to_cut > len(sequences[0]):
        words_to_cut -= len(sequences[0])
        sequences = sequences[1:]
    
    sequences[0] = sequences[0][words_to_cut:]
    return sequences


def truncate_sequences_dual(sequences, max_length):
    words_to_cut = sum(list(map(len, sequences))) - max_length
    if words_to_cut <= 0:
        return sequences

    words_to_cut_before = math.ceil(words_to_cut / 2.0)
    words_to_cut_after = words_to_cut // 2

    while words_to_cut_before > len(sequences[0]):
        words_to_cut_before -= len(sequences[0])
        sequences = sequences[1:]
    sequences[0] = sequences[0][words_to_cut_before:]

    while words_to_cut_after > len(sequences[-1]):
        words_to_cut_after -= len(sequences[-1])
        sequences = sequences[:-1]
    last = len(sequences[-1]) - words_to_cut_after
    sequences[-1] = sequences[-1][:last]

    return sequences


def write_linking_preds(output_file, data_infos, pred_ids, pred_scores):
    # Flatten the data_infos
    data_infos = [
        {"context_id": info["context_ids"][i], "turn_id": info["turn_ids"][i], "fact_id": info["fact_ids"][i]}
        for info in data_infos
        for i in range(len(info["context_ids"]))
    ]

    labels = []
    # Update the dialogs with linking result
    for pid, info in enumerate(data_infos):
        pred_id = pred_ids[pid]
        pred_score = pred_scores[pid]
        assert len(pred_score) == 2

        labels.append({"context_id": info["context_id"], "turn_id": info["turn_id"],
                       "fact_id": info["fact_id"], "target": bool(pred_id),
                       "score": [float(pred_score[0]), float(pred_score[1])]})

    if os.path.dirname(output_file) and not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    with open(output_file, "w") as jsonfile:
        logger.info("Writing predictions to {}".format(output_file))
        json.dump(labels, jsonfile, indent=2)


def write_generation_preds(dataset_walker, output_file, responses, infos):

    outputs = []
    for i in range(len(dataset_walker)):
        outputs.append(deepcopy({"text": None, "context_id": None, "turn_id": None}))

    for rid, response in enumerate(responses):
        outputs[rid]["text"] = response
        outputs[rid]["context_id"] = infos["context_ids"][rid]
        outputs[rid]["turn_id"] = infos["turn_ids"][rid]
        if infos.get("head_ids"):
            outputs[rid]["head_id"] = infos["head_ids"][rid]
        else:
            outputs[rid]["head_id"] = -1
        if infos.get("fact_ids"):
            outputs[rid]["fact_id"] = infos["fact_ids"][rid]
        else:
            outputs[rid]["fact_id"] = -1

    if os.path.dirname(output_file) and not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    with open(output_file, "w") as jsonfile:
        logger.info("Writing predictions to {}".format(output_file))
        json.dump(outputs, jsonfile, indent=2)
