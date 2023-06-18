import os
import json
from kogito.models.bart.comet import *

# prepare data in csv/tsv format directly
path = ''
for data_type in ['train', 'val', 'test']:
    data_file = os.path.join(path, 'data_persona_gen', f'neural_kg_data_{data_type}.json')
    with open(data_file, 'r') as f:
        data = json.load(f)
    with open(data_file.split('.')[0]+'.tsv', 'w') as f:
        print(data_file.split('.')[0]+'.tsv')
        f.writelines('\n'.join(data) + '\n')

# train comet-bart
config = COMETBARTConfig(
   output_dir=os.path.join(path, "bart"),
   num_workers=2,
   learning_rate=1e-5,
   gpus=1,
   sortish_sampler=True,
   atomic=True,
   max_epochs=20,
   pretrained_model="facebook/bart-large",
)
model = COMETBART(config)
model.tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, "comet-bart-ai2"))

train_graph = KnowledgeGraph.from_csv(
   os.path.join(path, 'data_persona_gen', "neural_kg_data_train.tsv"),
   sep="\t", header=None)
val_graph = KnowledgeGraph.from_csv(
   os.path.join(path, 'data_persona_gen', "neural_kg_data_val.tsv"),
   sep="\t", header=None)
test_graph = KnowledgeGraph.from_csv(
   os.path.join(path, 'data_persona_gen', "neural_kg_data_test.tsv"),
   sep="\t", header=None)

model.train(train_graph=train_graph, val_graph=val_graph, test_graph=test_graph)
model.evaluate(test_graph, 
   metrics=["bleu", "rouge", "cider", "meteor", "bert-score"], 
   top_k=5, batch_size=256)

# Save as a pretrained model
model.save_pretrained(os.path.join(path, 'runs', "comet-bart/v1"))