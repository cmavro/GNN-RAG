import argparse
import sys


def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_shared_args(parser):
    parser.add_argument('--name', default='webqsp', type=str)
    parser.add_argument('--data_folder', default='data/webqsp/', type=str)
    parser.add_argument('--max_train', default=200000, type=int)

    # embeddings
    parser.add_argument('--word2id', default='vocab.txt', type=str)
    parser.add_argument('--relation2id', default='relations.txt', type=str)
    parser.add_argument('--entity2id', default='entities.txt', type=str)
    parser.add_argument('--char2id', default='chars.txt', type=str)
    parser.add_argument('--entity_emb_file', default=None, type=str)
    parser.add_argument('--relation_emb_file', default=None, type=str)
    parser.add_argument('--relation_word_emb', default=True, type=bool_flag)
    parser.add_argument('--word_emb_file', default='word_emb.npy', type=str)
    parser.add_argument('--rel_word_ids', default='rel_word_idx.npy', type=str)
    parser.add_argument('--kge_frozen', default=0, type=int)
    parser.add_argument('--lm', default='lstm', type=str, choices=['lstm', 'bert', 'roberta', 'sbert', 't5','sbert2', 'dbert', 'simcse', 'relbert'])
    parser.add_argument('--lm_frozen', default=1, type=int)

    # dimensions, layers, dropout
    parser.add_argument('--entity_dim', default=50, type=int)
    parser.add_argument('--kg_dim', default=100, type=int)
    parser.add_argument('--word_dim', default=300, type=int)
    parser.add_argument('--lm_dropout', default=0.3, type=float)
    parser.add_argument('--linear_dropout', default=0.2, type=float)

    # optimization
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--warmup_epoch', default=0, type=int)
    parser.add_argument('--fact_scale', default=3, type=int)
    parser.add_argument('--eval_every', default=2, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--gradient_clip', default=1.0, type=float)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--decay_rate', default=0.0, type=float)
    parser.add_argument('--seed', default=19960626, type=int)
    parser.add_argument('--lr_schedule', action='store_true')
    parser.add_argument('--label_smooth', default=0.1, type=float)
    parser.add_argument('--fact_drop', default=0, type=float)
    #parser.add_argument('--encode_type', action='store_true')

    # model options

    parser.add_argument('--is_eval', action='store_true')
    parser.add_argument('--checkpoint_dir', default='checkpoint/pretrain/', type=str)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--experiment_name', default='', type=str)
    parser.add_argument('--load_experiment', default=None, type=str)
    parser.add_argument('--load_ckpt_file', default=None, type=str)
    parser.add_argument('--eps', default=0.95, type=float) # threshold for f1
    parser.add_argument('--test_batch_size', default=20, type=int)
    parser.add_argument('--q_type', default='seq', type=str)



def add_parse_args(parser):
    
    subparsers = parser.add_subparsers(help='Reason KGQA model')

    parser_rearev = subparsers.add_parser("ReaRev")
    create_parser_rearev(parser_rearev)

    parser_nsm = subparsers.add_parser("NSM")
    create_parser_nsm(parser_nsm)

    parser_graftnet = subparsers.add_parser("GraftNet")
    create_parser_graftnet(parser_graftnet)

    parser_nutrea = subparsers.add_parser("NuTrea")
    create_parser_nutrea(parser_nutrea)


def create_parser_rearev(parser):

    parser.add_argument('--model_name', default='ReaRev', type=str, choices=['ReaRev'])
    parser.add_argument('--alg', default='bfs', type=str)
    parser.add_argument('--num_iter', default=2, type=int)
    parser.add_argument('--num_ins', default=3, type=int)
    parser.add_argument('--num_gnn', default=3, type=int)
    parser.add_argument('--loss_type', default='kl', type=str)
    parser.add_argument('--use_self_loop', default=True, type=bool_flag)
    parser.add_argument('--normalized_gnn', default=False, type=bool_flag)
    parser.add_argument('--norm_rel', action='store_true')
    parser.add_argument('--data_eff', action='store_true')
    parser.add_argument('--pos_emb', action='store_true')
    add_shared_args(parser)


def create_parser_nsm(parser):
    parser.add_argument('--model_name', default='NSM', type=str, choices=['NSM'])
    parser.add_argument('--num_step', default=3, type=int)
    parser.add_argument('--reason_kb', default=False, type=bool_flag)
    parser.add_argument('--loss_type', default='kl', type=str)
    parser.add_argument('--lambda_constrain', default=0.0, type=float)
    parser.add_argument('--lambda_back', default=0.0, type=float)
    parser.add_argument('--use_self_loop', default=True, type=bool_flag)
    parser.add_argument('--use_inverse_relation', action='store_true')
    parser.add_argument('--norm_rel', action='store_true')
    parser.add_argument('--normalized_gnn', default=False, type=bool_flag)
    parser.add_argument('--data_eff', action='store_true')
    add_shared_args(parser)

def create_parser_graftnet(parser):
    parser.add_argument('--model_name', default='GraftNet', type=str, choices=['GraftNet'])
    parser.add_argument('--pagerank_lambda', default=0.8, type=float)
    parser.add_argument('--loss_type', default='bce', type=str)
    parser.add_argument('--num_layer', default=3, type=int)
    parser.add_argument('--use_inverse_relation', action='store_true')
    parser.add_argument('--norm_rel', action='store_true')
    parser.add_argument('--normalized_gnn', default=False, type=bool_flag)
    parser.add_argument('--data_eff', action='store_true')
    #parser.add_argument('--use_self_loop', default=True, type=bool_flag)
    add_shared_args(parser)
