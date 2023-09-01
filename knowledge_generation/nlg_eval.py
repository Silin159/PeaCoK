import json
from tqdm import tqdm
import argparse
import numpy as np
from nlgeval import NLGEval


def main():
    """
    Automation NLG evaluation script.
    """
    parser = argparse.ArgumentParser(description="Arguments for NLG evaluation.")
    parser.add_argument(
        "--hypothesis_path",
        type=str,
        help="Path for json file of hypotheses.")
    parser.add_argument(
        "--references_path",
        type=str,
        help="Path for json file of references (doesn't have to be ordered).")
    parser.add_argument(
        "--results_path",
        type=str,
        help="Path for json file of average nlg metric results.")
    args = parser.parse_args()

    with open(args.hypothesis_path, mode='r', encoding='utf-8') as f:
        hypotheses = json.load(f)
        f.close()

    with open(args.references_path, mode='r', encoding='utf-8') as f:
        references = json.load(f)
        f.close()
    
    # 1) Gather the references in the order of the hypotheses
    head_rel_2_references_dict = {}
    for str_instance in references:
        str_list = str_instance.split('\t')
        head = str_list[0]
        rel = str_list[1]
        head_rel_key = "{}\t{}".format(head, rel)
        if head_rel_key not in head_rel_2_references_dict.keys():
            head_rel_2_references_dict[head_rel_key] = []        
        head_rel_2_references_dict[head_rel_key].append(str_instance)

    formatted_references = []
    for str_instance in hypotheses:
        str_list = str_instance.split('\t')
        head = str_list[0]
        rel = str_list[1]
        head_rel_key = "{}\t{}".format(head, rel)
        formatted_references.append(head_rel_2_references_dict[head_rel_key])

    # 2) Evaluation on generation metrics
    nlgeval = NLGEval()
    # NOTE: you can skip skipthoughts by uncommenting the next line
    #       if you want the eval to be quick
    nlgeval.no_skipthoughts = True
    nlgeval.no_glove = True
    # nlgeval.no_overlap = True

    overall_scores_dict = {}
    assert len(hypotheses) == len(formatted_references)
    eval_loop = tqdm(list(zip(hypotheses, formatted_references)), desc='eval: ')
    for hypo, refs in eval_loop:
        assert len(refs) != 0
        scores_dict = nlgeval.compute_individual_metrics(ref=refs, hyp=hypo)
        for k in scores_dict.keys():
            if k not in overall_scores_dict:
                overall_scores_dict[k] = []
            overall_scores_dict[k].append(scores_dict[k])

    # 3) Calculate the average and save
    avg_score_dict = {score_name: np.mean(np.array(score_list)) for score_name, score_list in overall_scores_dict.items()}
    print(avg_score_dict)
    avg_score_dict = {score_name: float(avg_score) for score_name, avg_score in  avg_score_dict.items()}

    with open(args.results_path,  mode='w', encoding='utf-8') as f:
        json.dump(avg_score_dict, f, indent=4)


if __name__ == '__main__':
    main()