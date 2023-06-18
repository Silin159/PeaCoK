from parlai.scripts.eval_model import eval_model, setup_args as base_setup_args

IS_ORIGINAL = False


def setup_task():
    if IS_ORIGINAL:
        task_name = 'tasks.convai2transmitter.agents:SelfOriginalTeacher'
    else:
        task_name = 'tasks.convai2transmitter.agents:SelfRevisedTeacher'
    return task_name


def setup_trained_weights():
    # /mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/sigao/p2bot_ext/checkpoint/psquare_revised.model
    if IS_ORIGINAL:
        weights_name = '/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/sigao/p2bot_ext/psquare_pf_original/psquare_original.model'
    else:
        weights_name = '/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/sigao/p2bot_ext/psquare_pf_revised/psquare_revised.model'
    return weights_name


def setup_args(parser=None):
    parser = base_setup_args(parser)
    task_name = setup_task()
    parser.set_defaults(
        task=task_name,
        datatype='valid',
        hide_labels=False,
        metrics='hits@1',
    )
    return parser


def eval_hits(opt, print_parser):
    report = eval_model(opt, print_parser)
    print('============================')
    print('FINAL Hits@1: ' + str(report['hits@1']))


if __name__ == '__main__':
    parser = setup_args()
    model_name = setup_trained_weights()
    parser.set_params(
        model='agents.transmitter.transmitter:TransformerAgent',  # agents.psquare.psquare:PSquareAgent
        model_file=model_name,
        init_model_transmitter=model_name,
        gpu=1,
        batchsize=16,
        beam_size=1,
        rank_candidates=True,
        report_freq=0.0001,
    )
    print("Model: {}".format(model_name))
    opt = parser.parse_args(print_args=False)
    eval_hits(opt, print_parser=parser)
