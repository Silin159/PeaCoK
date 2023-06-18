import torch
# import torch.nn.functional as F
import logging
import numpy as np

logger = logging.getLogger(__name__)


def softmax(array, axis=1):
    e_x = np.exp(array)
    sum_axis = np.sum(e_x, axis=axis, keepdims=True)
    return e_x / sum_axis


def run_batch_linking(args, model, batch, tokenizer=None):
    # batch = tuple(input_tensor.to(args.device) for input_tensor in batch)# if isinstance(input_tensor, torch.Tensor))
    # input_ids, labels = batch

    model_outputs = model(input_ids=batch['input_ids'], labels=batch['labels'])

    cls_loss = model_outputs[0]
    cls_logits = model_outputs[1]
    lm_logits = None

    return cls_loss, lm_logits, cls_logits, batch['labels']


def run_batch_generation_train(args, model, batch, tokenizer=None):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, decoder_input_ids, lm_label_ids = batch

    outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=lm_label_ids)

    gen_loss = outputs[0]

    return gen_loss


def run_batch_generation_eval(args, model, batch, tokenizer):
    batch_t = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids = batch_t

    batch_ut = tuple(input_tensor for input_tensor in batch if not isinstance(input_tensor, torch.Tensor))
    lm_target_text, data_info = batch_ut

    batch_size = input_ids.size(0)
    k_sample = 100
    p_sample = 0.95
    num_seq = 5

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):  # validation
        gen_ids = model.module.generate(input_ids=input_ids, use_cache=True, min_length=1, max_length=25,
                                        do_sample=True, top_k=k_sample, top_p=p_sample, num_return_sequences=num_seq)
    else:  # test
        gen_ids = model.generate(input_ids=input_ids, use_cache=True, min_length=1, max_length=25,
                                 do_sample=True, top_k=k_sample, top_p=p_sample, num_return_sequences=num_seq)

    gen_text = []
    for split in range(0, batch_size * num_seq, num_seq):
        gen_text_single = tokenizer.batch_decode(gen_ids[split:(split+num_seq), :], skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=True)
        gen_text_single = list(map(str.strip, gen_text_single))
        gen_text.append(gen_text_single)

    return gen_text, lm_target_text, data_info
