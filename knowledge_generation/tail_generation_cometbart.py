import os
import torch
import argparse
import json
import jsonlines

from tqdm import tqdm
from kogito.core.utils import (
    chunks,
    trim_batch
)

from kogito.core.knowledge import KnowledgeGraph
from kogito.models.bart.comet import COMETBART

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_seed(seed):
    """ Function to set torch seed. """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def our_generate(
        comet_model,
        input_graph: KnowledgeGraph,
        decode_method: str = "greedy",
        num_beams: int = 1,
        num_generate: int = 3,
        batch_size: int = 64,
        max_length: int = 24,
        min_length: int = 1,
    ) -> KnowledgeGraph:
    """
    Generate inferences with the Comet model with extra generate parameters.
    """
    with torch.no_grad():
        outputs = []
        for kg_batch in tqdm(list(chunks(input_graph, batch_size))):
            queries = []
            for kg_input in kg_batch:
                queries.append(kg_input.to_query(decode_method=decode_method))
            batch = comet_model.tokenizer(
                queries, return_tensors="pt", truncation=True, padding="max_length"
            ).to(device)
            input_ids, attention_mask = trim_batch(
                **batch, pad_token_id=comet_model.tokenizer.pad_token_id
            )

            if decode_method == "greedy":
                num_generate = 1
                summaries = comet_model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=comet_model.config.decoder_start_token_id,
                    num_beams=1, # NOTE: this means no beam search, just greedy
                    num_return_sequences=1,
                    max_new_tokens=max_length,
                    min_length=min_length,
                    do_sample=False
                )
            elif decode_method == "beam":
                summaries = comet_model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=comet_model.config.decoder_start_token_id,
                    num_beams=num_beams,
                    num_return_sequences=num_generate,
                    max_new_tokens=max_length,
                    min_length=min_length,
                    do_sample=False
                )
            elif decode_method == "top-p":
                summaries = comet_model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=comet_model.config.decoder_start_token_id,
                    num_return_sequences=num_generate,
                    max_new_tokens=max_length,
                    min_length=min_length,
                    do_sample=True,
                    top_p=0.6,
                    top_k=0
                )

            output = comet_model.tokenizer.batch_decode(
                summaries,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for kg_input, generations in zip(
                kg_batch, list(chunks(output, num_generate))
            ):
                output_kg = kg_input.copy()
                output_kg.tails = generations
                outputs.append(output_kg)

        return KnowledgeGraph(outputs)


def main():
    """
    Script to generate and save COMETBART generations.
    """
    load_seed(84)

    parser = argparse.ArgumentParser(description="Arguments for COMETBART generation.")
    parser.add_argument(
        "--cometbart_inputs_path",
        type=str,
        default=os.path.join("data_persona_gen", "neural_kg_data_test_unique.jsonl"),
        help="jsonl file for kogito knowledge graph with unique head/rel")
    parser.add_argument(
        "--cometbart_output_path",
        type=str,
        default=os.path.join("data_persona_gen", "cometbart_kg_data"),
        help="Path for json file (of head/rel/tails) to save COMETBART generated tails.")
    parser.add_argument(
        "--cometbart_model_name",
        type=str,
        default=os.path.join("model_checkpoints", "bart_epoch07", "best_tfmr"),
        help="CometBART huggingface or local model name to be loaded with from_pretrained()."
    )
    args = parser.parse_args()

    # 1) Load input KG
    kg_graph = KnowledgeGraph.from_jsonl(args.cometbart_inputs_path)

    # 2) Load trained model
    model = COMETBART.from_pretrained(args.cometbart_model_name)
    print(model.config.task)
    
    # 3) Generate the tails for the graph heads and rels
    generated_graph_beam = our_generate(
        comet_model=model,
        input_graph=kg_graph,
        decode_method="beam",
        num_generate=5,
        num_beams=5,
        batch_size=64,
        max_length=12,
        min_length=1,
    )

    # 4) Save as JSON instead of JSONL for evaluation script
    generated_graph_beam.to_jsonl(args.cometbart_output_path + ".jsonl")
    all_json_entries = []
    with open(args.cometbart_output_path + ".jsonl", mode='r', encoding='utf-8') as f:
        for instance in jsonlines.Reader(f):
            head = instance["head"]
            rel = instance["relation"]
            for tail in instance["tails"]:
                json_entry = "{}\t{}\t{}".format(head, rel, tail) 
                all_json_entries.append(json_entry)

    with open(args.cometbart_output_path + ".json", mode='w', encoding='utf-8') as f:
        json.dump(all_json_entries, f, ensure_ascii=False, indent=4)
        f.close()

    print("Comet-BART generations saved at {}.jsonl and {}.json".format(args.cometbart_output_path, args.cometbart_output_path))

if __name__ == '__main__':
    main()