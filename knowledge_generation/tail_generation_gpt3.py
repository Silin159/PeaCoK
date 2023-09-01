import os
import argparse
import json
import jsonlines
import openai
from tqdm import tqdm
from collections import OrderedDict
import random


def load_seed(seed):
    """ Function to set random seed. """
    random.seed(seed)


def main():
    """
    Script to generate and save GPT-3 generations.
    """
    load_seed(84)
    
    parser = argparse.ArgumentParser(description="Arguments for GPT-3 generation.")
    parser.add_argument(
        "--incontext_examples_path",
        type=str,
        default=os.path.join("data_persona_gen", "neural_kg_data_train.json"),
        help="")
    parser.add_argument(
        "--gpt3_inputs_path",
        type=str,
        default=os.path.join("data_persona_gen", "neural_kg_data_test.json"),
        help="Path for json file of inputs to generate new GPT-3 tails.")
    parser.add_argument(
        "--gpt3_dict_path",
        type=str,
        default=os.path.join("data_persona_gen", "gpt3_output_dicts.jsonl"),
        help="Path for jsonlines file to save GPT-3 output dictionaries.")
    parser.add_argument(
        "--gpt3_text_path",
        type=str,
        default=os.path.join("data_persona_gen", "gpt3_kg_data.json"),
        help="Path for json file (of head/rel/tails) to save new GPT-3 tails.")
    parser.add_argument(
        "--gpt3_top1_text_path",
        type=str,
        default=os.path.join("data_persona_gen", "gpt3_top1picks.json"),
        help="Path for json file (of head/rel/tails) to only save best GPT-3 tails according to OpenAI.")
    parser.add_argument(
        "--k_shot",
        type=int,
        default=5,
        help="Few-shot k.")
    parser.add_argument(
        "--gpt3_model_name",
        type=str,
        default="text-davinci-003",
        help="GPT-3 OpenAI API model name.")
    parser.add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Flag to make script verbose.")
    args = parser.parse_args()

    # 1) Load the in-context examples and gather them by head and rel, save as dict
    with open(args.incontext_examples_path, mode='r', encoding='utf-8') as f:
        incontext_examples = json.load(f)
        f.close()
   
    rel_to_head_to_tail_dict = OrderedDict()
    for str_instance in incontext_examples:
        str_list = str_instance.split('\t')
        head = str_list[0]
        rel = str_list[1]
        tail = str_list[2]
        head_rel_key = "{}\t{}".format(head, rel)
        if rel not in rel_to_head_to_tail_dict.keys():
            rel_to_head_to_tail_dict[rel] = OrderedDict()
        if head not in rel_to_head_to_tail_dict[rel].keys():
            rel_to_head_to_tail_dict[rel][head] = tail

    # 2) Gather relation-specific generation instructions 
    rel_instruct_dict = {
        "characteristic": "here is my character trait:",
        "routine_habit": "here is what I regularly or consistently do:",
        "goal_plan": "here is what I will do or achieve in the future:",
        "experience": "here is what I did in the past:",
        "characteristic_relationship": "here is my character trait related to other people or social groups:",
        "routine_habit_relationship": "here is what I regularly or consistently do related to other people or social groups:",
        "goal_plan_relationship": "here is what I will do or achieve in the future related to other people or social groups:",
        "experience_relationship": "here is what I did in the past related to other people or social groups:"
    }

    # 3) Load the tuples and separate head/rels that we need to generate for
    with open(args.gpt3_inputs_path, mode='r', encoding='utf-8') as f:
        input_tuples = json.load(f)
        f.close()
    input_queries = set()
    for str_instance in input_tuples:
        str_list = str_instance.split('\t')
        head = str_list[0]
        rel = str_list[1]
        head_rel_key = "{}\t{}".format(head, rel)
        input_queries.add(head_rel_key)
    input_queries = list(input_queries)
    input_queries.sort()
    print("Generating for {} queries...".format(len(input_queries)))
    
    # NOTE: uncomment if testing the script // only 3 generations
    #       so we don't request too much at once from OpenAI API
    input_queries = input_queries[:3]

    # 4) Request generation from OpenAI
    gpt3_gen_text = []
    gpt3_gen_dict_list = []
    query_loop = tqdm(input_queries, desc='generation: ')
    openai.api_key = args.openai_api_key

    for input_query in query_loop:
        # a) get the context
        if args.verbose:
            print("-" * 50)
            print("Input query: ", input_query)
        str_list = input_query.split('\t')
        head = str_list[0]
        rel = str_list[1]
        context_heads = []
        contexts = []
        input_prompt = ""
        # b) attach to input query as a prompt
        if args.k_shot == 0:
            input_prompt = head  + ", " + rel_instruct_dict[rel] + " "
        else:
            context_heads = [h for h in rel_to_head_to_tail_dict[rel].keys() if h != head]
            context_heads = random.sample(population=context_heads, k=args.k_shot)
            if args.verbose:
                print("Sampled context heads: ", context_heads)
            contexts = [h + "\t" + rel + "\t" + rel_to_head_to_tail_dict[rel][h] + "\n\n" for h in context_heads]
            input_prompt = ('').join(contexts) + head  + "\t" + rel + "\t"
        
        if args.verbose:
            print("Input prompt:")
            print(input_prompt)
        
        # c) get the generation from open AI
        result = openai.Completion.create(
            model=args.gpt3_model_name, 
            prompt=input_prompt, 
            max_tokens=12, 
            top_p=1.0,
            n=5, 
            temperature=0.9
        )
        result["input_prompt"] = input_prompt
        result["context_heads"] = context_heads
        result["contexts"] = contexts
        result["head"] = head
        result["rel"] = rel
        if args.k_shot == 0: # GPT-3.5 processing
            tail_list_list = [output_dict["text"].split("\n") for output_dict in result["choices"]]
            tails = []
            for tail_list in tail_list_list:
                inner_list = []
                for tail in tail_list:
                    tail = tail.strip().strip("-").strip("1.").strip().replace("\t", " ")
                    if tail != "":
                        inner_list.append(tail)
                if len(inner_list) == 0:
                    tails.append("")
                else:
                    tails.append(inner_list[0])
        else: # GPT-3 processing
            tail_list_list = [output_dict["text"].split("\n") for output_dict in result["choices"]]
            tails = []
            for tail_list in tail_list_list:
                inner_list = []
                for tail in tail_list:
                    tail = tail.strip()
                    if tail != "":
                        inner_list.append(tail)
                if len(inner_list) == 0:
                    tails.append("")
                else:
                    tails.append(inner_list[0])
        assert len(tails) == 5
        result["tails"] = tails

        gpt3_gen_dict_list.append(result)
        gpt3_gen_text.extend(
            [head  + "\t" + rel + "\t" + tail for tail in tails]
        )
    
        # d) save the output dicts and the data in head \t rel \t tail format
        with open(args.gpt3_dict_path, mode='w', encoding='utf-8') as f:
            writer = jsonlines.Writer(f)
            writer.write_all(gpt3_gen_dict_list)
        writer.close()
        f.close()

        with open(args.gpt3_text_path, mode='w', encoding='utf-8') as f:
            json.dump(gpt3_gen_text, f, ensure_ascii=False, indent=4)
            f.close()

    # 5) Pick first instance of head rel only
    seen_head_sel = []
    top1pick_list = []

    for gen_string in gpt3_gen_text:
        head_rel_key = gen_string.split("\t")[:2]
        if head_rel_key not in seen_head_sel:
            top1pick_list.append(gen_string)
            seen_head_sel.append(head_rel_key)
    
    # 6) Save this version as GPT's top 1 pick
    with open(args.gpt3_top1_text_path, mode='w', encoding='utf-8') as f:
        json.dump(top1pick_list, f, ensure_ascii=False, indent=4)
        f.close()
    
    print("GPT-3 generations saved at {} and {}.".format(args.gpt3_text_path, args.gpt3_top1_text_path))

                                                                                                                               
if __name__ == '__main__':
    main()