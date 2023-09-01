import os
import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import DebertaV2ForSequenceClassification, AutoTokenizer


class KnowledgeGraphDataset(Dataset):
    """
    A Dataset class that manages the KG.
    """
    def __init__(self, knowledge_list, lm_name='microsoft/deberta-v3-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        head_rel_cnt = 0
        tail_cnt = 0
        self.head_rel_dict = {}
        self.head_rel_list = []
        self.tail_list = []
        self.input_strs = []

        for str_instance in knowledge_list:
            str_list = str_instance.split('\t')
            head = str_list[0]
            rel = str_list[1]
            tail = str_list[2]
            head_rel_key = "{}\t{}".format(head, rel)
            if head_rel_key not in self.head_rel_dict.keys():
                self.head_rel_dict[head_rel_key] = {}
                self.head_rel_dict[head_rel_key]["head"] = head
                self.head_rel_dict[head_rel_key]["rel"] = rel
                self.head_rel_dict[head_rel_key]["tails"] = []
                self.head_rel_dict[head_rel_key]["tail_relevancy"] = []
                head_rel_cnt += 1
            
            self.head_rel_dict[head_rel_key]["tails"].append(tail)
            self.input_strs.append(str_instance)
            self.tail_list.append(tail)
            self.head_rel_list.append(head_rel_key)
            tail_cnt += 1
        
        print("Gathered {} head-rel pairs and {} tails in total.".format(head_rel_cnt, tail_cnt))
        self.inputs = (self.tokenizer(self.input_strs, return_tensors='pt', padding=True, truncation=True))
        

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.inputs.items()}
        item['input_str'] = self.input_strs[idx]
        item['head_rel'] = self.head_rel_list[idx]
        item['tail'] = self.tail_list[idx]
        return item

    def __len__(self):
        return len(self.input_strs)


def load_seed(seed):
    """ Function to set torch seed. """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def deberta_eval(deberta_model, kg_dataloader, kg_dataset, use_cuda=False):
    """
    Retrieves DeBERTa relevancy score and picks the top ranking tail.

    Args:
        deberta_model: DeBERTa model that ranks the tails.
        kg_dataloader: Dataloader of the KG we are ranking tails for.
        kg_dataset: Dataset of the KG we are ranking tails for.
        use_cuda (bool, optional): Option to load onto GPU. Defaults to False.

    Returns:
        The overall relevancy score and the list of top 1 picks.
    """
    
    test_loop = tqdm(kg_dataloader, desc='eval: ')
    deberta_model.eval()
    if use_cuda:
        deberta_model.cuda()
    relevancy_sum = 0.0
    top1pick_list = []
    tot_seen = 0.0
    logit_tracker = []

    for batch in test_loop:
        input_ids = batch['input_ids'].detach()
        attention_mask = batch['attention_mask'].detach()
        head_rel_keys = batch['head_rel']
        tails = batch['tail']

        if use_cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        with torch.no_grad():
            output = deberta_model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            predicted_class_id = logits.argmax(dim=-1)
            tot_seen += len(input_ids)
            relevancy_sum += predicted_class_id.sum().item()
            logit_tracker.extend(list(zip(head_rel_keys, tails, logits[:,1].cpu().numpy())))

    relevancy_score = float(relevancy_sum) / tot_seen

    for head_rel_key, tail, relevancy in logit_tracker:
        kg_dataset.head_rel_dict[head_rel_key]["tail_relevancy"].append((relevancy, tail))

    for head_rel_key in kg_dataset.head_rel_dict.keys():
        # print("____")
        # print("head_rel: ", head_rel_key)
        relevant_tails = kg_dataset.head_rel_dict[head_rel_key]["tail_relevancy"]
        most_relevant_tail = max(relevant_tails)[1]
        # print("relevant tail list: ", relevant_tails)
        # print("most: ", most_relevant_tail)
        top1pick_list.append("{}\t{}".format(head_rel_key, most_relevant_tail))

    return relevancy_score, top1pick_list


def model_and_data_loading(args):
    """
    Loads model and dataloader.
    Returns model, dataloader, dataset
    """
    # NOTE: the id2label, config.id2label = {0: 'LABEL_0', 1: 'LABEL_1'}
    deberta_model = DebertaV2ForSequenceClassification.from_pretrained(args.deberta_checkpoint_path)

    with open(args.generated_instances_path, mode='r', encoding='utf-8') as f:
        generated_instances = json.load(f)
    
    # print(type(generated_instances))
    # print(len(generated_instances))
    
    kg_dataset = KnowledgeGraphDataset(knowledge_list=generated_instances)
    kg_loader = DataLoader(
        kg_dataset, 
        batch_size=args.eval_batch_size, 
        shuffle=False
    )
    return deberta_model, kg_loader, kg_dataset


def main():
    """
    Script to rank generations with DeBERTa, and pick the top scoring one.
    """
    load_seed(84)

    parser = argparse.ArgumentParser(description="Arguments for DeBERTa evaluation.")
    parser.add_argument(
        "--deberta_checkpoint_path",
        type=str,
        default=os.path.join("model_checkpoints", "deberta-checkpoint-97284"),
        help="DeBERTa checkpoint path to load it.")
    parser.add_argument(
        "--generated_instances_path",
        type=str,
        help="Path for json file of generated head/rel/list of tail entries.")
    parser.add_argument(
        "--deberta_top_choice_path",
        type=str,
        help="Path for json file of to save the top-1 generations according to DeBERTa.")
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Evaluation batch size.")
    args = parser.parse_args()

    deberta_model, kg_loader, kg_dataset = model_and_data_loading(args)

    use_cuda = torch.cuda.is_available()
    relevancy_score, top1pick_list = deberta_eval(deberta_model, kg_loader, kg_dataset, use_cuda)
    print("Generated tuples received overall relevancy score of: ", relevancy_score)
    
    with open(args.deberta_top_choice_path, mode='w', encoding='utf-8') as f:
        json.dump(top1pick_list, f, ensure_ascii=False, indent=4)
        f.close()

    print("Top picks saved at {}.".format(args.deberta_top_choice_path))


if __name__ == '__main__':
    main()
