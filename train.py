import os
import torch
import random
import numpy as np

from core.train_options import TrainOptions
from core.cfgs import cfg, parse_args_extend
from utils.train_utils import prepare_env
from core.trainer import Trainer

import torch.distributed as dist

import logging
logger = logging.getLogger(__name__)


def main(gpu, ngpus_per_node, options):
    parse_args_extend(options)

    options.batch_size = cfg.TRAIN.BATCH_SIZE
    options.workers = cfg.TRAIN.NUM_WORKERS

    options.gpu = gpu
    options.ngpus_per_node = ngpus_per_node

    if options.distributed:
        dist.init_process_group(backend=options.dist_backend, init_method=options.dist_url,
                                world_size=options.world_size, rank=options.local_rank)

    if options.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        options.rank, world_size = dist.get_rank(), dist.get_world_size()
        assert options.rank == options.local_rank
        assert world_size == options.world_size

    trainer = Trainer(options)
    trainer.fit()


if __name__ == '__main__':
    options = TrainOptions().parse_args()
    parse_args_extend(options)
    if options.local_rank == 0:
        prepare_env(options)
    else:
        options.checkpoint_dir = ''

    if cfg.SEED_VALUE >= 0:
        logger.info(f'Seed value for the experiment {cfg.SEED_VALUE}')
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)

    ngpus_per_node = torch.cuda.device_count()
    options.distributed = (ngpus_per_node > 1) or options.multiprocessing_distributed
    if options.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        options.world_size = ngpus_per_node * options.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        # mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, options))
        main(options.local_rank, ngpus_per_node, options)
    else:
        # Simply call main_worker function
        # main_worker(args.gpu, ngpus_per_node, args)
        main(None, ngpus_per_node, options)
