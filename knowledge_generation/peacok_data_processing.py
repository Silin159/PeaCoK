import os
import json
import jsonlines
import random
from tqdm import tqdm


def get_unique_tails(data):
    tails = {}
    for l in data:
        head, relation, tail = l.split('\t')
        if relation not in tails:
            tails[relation] = set()
        tails[relation].add(tail)

    return tails


def populate_tails(data):
    tails = {'train': [], 'val': [], 'test': []}
    tails['train'] = get_unique_tails(data['train'])

    tails['val'] = get_unique_tails(data['val'])
    for r in tails['val']:
        if r in tails['train']:
            tails['val'][r].union(tails['train'][r])
    for r in tails['train']:
        if r not in tails['val']:
            tails['val'][r] = tails['train'][r]

    tails['test'] = get_unique_tails(data['test'])
    for r in tails['test']:
        if r in tails['val']:
            tails['test'][r].union(tails['val'][r])
    for r in tails['val']:
        if r not in tails['test']:
            tails['test'][r] = tails['val'][r]

    return tails

def get_true_tails(data):
    true_tails = {}
    for data_type in data:
        for l in data[data_type]:
            head, relation, tail = l.split('\t')

            if head not in true_tails:
                true_tails[head] = {}
            if relation not in true_tails[head]:
                true_tails[head][relation] = set()

            true_tails[head][relation].add(tail)
    return true_tails

            

random.seed(42)

path = 'data_persona_gen/'
data = {'train': [], 'val': [], 'test': []}

for data_type in ['train', 'val', 'test']:
    data_file = os.path.join(path, f'neural_kg_data_{data_type}.json')
    with open(data_file, 'r') as f:
        data[data_type] = json.load(f)

# get unique tails for each data "segmentation"
unique_tails = populate_tails(data)
for d_type in unique_tails:
    for rel in unique_tails[d_type]:
        unique_tails[d_type][rel] = list(unique_tails[d_type][rel])
with open(os.path.join(path, 'unique_tails.json'), 'w') as f:
    json.dump(unique_tails, f)

# with open(os.path.join(path, 'unique_tails.json'), 'r') as f:
#     unique_tails = json.load(f)   

# get legitimate tails for each head+relation 
true_tails = get_true_tails(data)
for head in true_tails:
    for rel in true_tails[head]:
      true_tails[head][rel] = list(true_tails[head][rel])
with open(os.path.join(path, 'true_tails.json'), 'w') as f:
    json.dump(true_tails, f)

# with open(os.path.join(path, 'true_tails.json'), 'r') as f:
#     true_tails = json.load(f)   

print('Generating first version of augmented data')
# augment data by generating equal number of bad triples
fake_data = {'train': [], 'val': [], 'test': []}
for data_type in ['train', 'val', 'test']:
    tails_in_use = {}
    for i in tqdm(range(len(data[data_type]))):
        l = data[data_type][i]
        head, relation, tail = l.split('\t')

        if head not in tails_in_use:
            tails_in_use[head] = {}
        if relation not in tails_in_use[head]:
            tails_in_use[head][relation] = []

        if len(unique_tails[data_type][relation]) <= (
                    len(tails_in_use[head][relation]) + 
                    len(true_tails[head][relation])):
            # no (more) possible fake combinations possible with our data
            continue

        candidate_tail = random.choice(unique_tails[data_type][relation])
        while candidate_tail in true_tails[head][relation] or \
                candidate_tail in tails_in_use[head][relation]:
            candidate_tail = random.choice(unique_tails[data_type][relation])

        tails_in_use[head][relation].append(candidate_tail)
        fake_data[data_type].append('\t'.join([head, relation, candidate_tail]))

# because we might have skipped some data points in the loop above,
# let's randomly augment data until we have the same number of items

for data_type in ['train', 'val', 'test']:
    print(f'{data_type}: OG={len(data[data_type])}; AD={len(fake_data[data_type])}')

for data_type in ['train', 'val', 'test']:
    while (len(fake_data[data_type]) < len(data[data_type])):

        l = random.choice(data[data_type])
        head, relation, tail = l.split('\t')

        if len(unique_tails[data_type][relation]) <= (
                    len(tails_in_use[head][relation]) + 
                    len(true_tails[head][relation])):
            # no (more) possible fake combinations possible with our data
            continue

        candidate_tail = random.choice(unique_tails[data_type][relation])
        while candidate_tail in true_tails[head][relation] or \
                candidate_tail in tails_in_use[head][relation]:
            candidate_tail = random.choice(unique_tails[data_type][relation])

        tails_in_use[head][relation].append(candidate_tail)
        fake_data[data_type].append('\t'.join([head, relation, candidate_tail]))


print('Saving data')
for data_type in ['train', 'val', 'test']:
    with open(os.path.join(path, f'fake_data_{data_type}.json'), 'w') as f:
        json.dump(fake_data[data_type], f)

# save unique head and relation pairs for COMETBART generation in jsonl format
for data_type in ['test']:
    path = "data_persona_gen"
    data_file = os.path.join(path, 'neural_kg_data_{}.json'.format(data_type))

    with open(data_file, mode='r', encoding='utf-8') as f:
        input_tuples = json.load(f)
        f.close()

    input_queries = []
    kg_dict_list = []
    for str_instance in input_tuples:
        str_list = str_instance.split('\t')
        head = str_list[0]
        rel = str_list[1]
        head_rel_key = "{}\t{}".format(head, rel)
        if head_rel_key not in input_queries:
            input_queries.append(head_rel_key)
            kg_dict_list.append({"head": head, "relation": rel, "tails": []})
    input_queries.sort()

    with open(os.path.join(path, 'neural_kg_data_{}_unique.jsonl'.format(data_type)), mode='w', encoding='utf-8') as f:
        writer = jsonlines.Writer(f)
        writer.write_all(kg_dict_list)
    writer.close()
    f.close()