import os
os.environ['PYTHONPATH'] = os.getcwd()
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '8'

import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a CMVPR in ddp')
    parser.add_argument(
        '--debug',
        dest='debug',
        action='store_true')
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to data storing file')
    parser.add_argument(
        '--weight_path',
        type=str,
        required=True,
        help='Path to weight saving file')
    parser.add_argument(
        '--resume_from',
        type=str,
        help='the model weights chose to load')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file')
    # parser.add_argument(
    #     '--need_eval',
    #     dest='need_eval',
    #     action='store_true')
    parser.add_argument(
        '--seed',
        type=int,
        default=3407,
        help='random seed')
    parser.add_argument(
        '-g',
        '--gpu',
        type=int,
        nargs='+',
        help='ids of gpus to use')
    parser.add_argument(
        '-p',
        '--master_port', default=29500, type=int,
        help="master node (rank 0)'s free port that needs to "
             "be used for communication during distributed training")
    args = parser.parse_args()
    return args


def args_to_str(args):
    argv = ['--config', args.config]
    if args.debug:
        argv.append('--debug')
    argv += ['--data_path', args.data_path]
    argv += ['--weight_path', args.weight_path]
    if args.resume_from is not None:
        argv += ['--resume_from', args.resume_from]
    # if args.need_eval:
    #     argv.append('--need_eval')
    argv += ['--gpu_num', str(len(args.gpu))]
    if args.seed is not None:
        argv += ['--seed', str(args.seed)]
    return argv


def main():
    args = parse_args()
    if args.gpu is not None:
        gpu = args.gpu
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu = [0]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu])
    from torch.distributed import launch
    # os.environ['training_script'] = './train.py'
    sys.argv = ['',
                '--nproc_per_node={}'.format(len(gpu)),
                '--master_port={}'.format(args.master_port),
                './train.py'
                ] + args_to_str(args) + ['--distributed']
    launch.main()


if __name__ == '__main__':
    main()