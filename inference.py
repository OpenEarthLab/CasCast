import argparse
import os
import torch
from utils.builder import ConfigBuilder
import utils.misc as utils
import yaml
from utils.logger import get_logger
import copy
from utils.metrics import MetricsRecorder


#----------------------------------------------------------------------------

def subprocess_fn(args):
    utils.setup_seed(args.seed * args.world_size + args.rank)

    logger = get_logger("test", args.run_dir, utils.get_rank(), filename=f'test_{args.start_timestamp}.log')

    # args.logger = logger
    args.cfg_params["logger"] = logger

    # build config
    logger.info('Building config ...')
    builder = ConfigBuilder(**args.cfg_params)

    # build model
    logger.info('Building models ...')
    model = builder.get_model()

    if model.use_ceph:
        ### TODO: warning ###
        # model_checkpoint = os.path.join(args.relative_checkpoint_dir, 'checkpoint_best.pth') 
        model_checkpoint = os.path.join(args.relative_checkpoint_dir, 'checkpoint_latest.pth')
    else:
        model_checkpoint = os.path.join(args.run_dir, 'checkpoint_best.pth')

    if not args.debug:
        model.load_checkpoint(model_checkpoint)
    model_without_ddp = utils.DistributedParallel_Model(model, args.local_rank)

    for key in model_without_ddp.model:
        params = [p for p in model_without_ddp.model[key].parameters() if p.requires_grad]
        cnt_params = sum([p.numel() for p in params])
        logger.info("params {key}: {params}".format(key=key, params=cnt_params))
        # print("params {key}:".format(key=key), cnt_params)


    # build dataset
    logger.info('Building dataloaders ...')
    
    dataset_params = args.cfg_params['dataset']

    test_dataloader = builder.get_dataloader(dataset_params=dataset_params, split = 'test', batch_size=args.batch_size)

    logger.info('valid dataloaders build complete')
    logger.info('begin valid ...')
    # model_without_ddp.test(test_data_loader=test_dataloader, epoch=0)
    
    model_without_ddp.eval_metrics = MetricsRecorder(args.metric_list)
    predict_len = test_dataloader.dataset.output_length
    if args.test_save_steps > 0 :
        model_without_ddp.checkpoint_savedir = os.path.join(args.relative_checkpoint_dir, args.start_timestamp)
    model_without_ddp.test_final(test_dataloader, predict_len)

def main(args):
    if args.world_size > 1:
        utils.init_distributed_mode(args)
    else:
        args.rank = 0
        args.local_rank = 0
        args.distributed = False
        args.gpu = 0
        torch.cuda.set_device(args.gpu)


    run_dir = args.cfgdir
    print(run_dir)
    
    # import pdb; pdb.set_trace()
    args.cfg = os.path.join(args.cfgdir, 'training_options.yaml')
    with open(args.cfg, 'r') as cfg_file:
        cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    # 根据申请的cpu数来设置dataloader的线程数
    cfg_params['dataloader']['num_workers'] = args.num_workers
    cfg_params['dataset']['test'] = copy.deepcopy(cfg_params['dataset']['valid'])
    # cfg_params['dataset']['test']['input_length'] = args.length
    cfg_params['dataset']['test']['sample_steps'] = args.steps
    cfg_params['dataset']['test']['test_period'] = {'start_timestamp': [args.start_timestamp], 'end_timestamp': [args.end_timestamp]}
    if "checkpoint_path" in cfg_params["model"]["params"]["extra_params"]:
        del cfg_params["model"]["params"]["extra_params"]["checkpoint_path"]

    #判断是否使用常量数据
    dataset_vnames = cfg_params['dataset']['train'].get("vnames", None)
    if dataset_vnames is not None:
        constants_len = len(dataset_vnames.get('constants'))
    else:
        constants_len = 0
    cfg_params['model']['params']['constants_len'] = constants_len
    cfg_params['model']['params']['extra_params']['test_save_steps'] = args.test_save_steps
    # cfg_params['model']['params'].pop('optimizer')
    # cfg_params['model']['params'].pop('lr_scheduler')

    if args.rank == 0:
        with open(os.path.join(run_dir, 'valid_options.yaml'), 'wt') as f:
            yaml.dump(vars(args), f, indent=2)
            yaml.dump(cfg_params, f, indent=2)

    args.cfg_params = cfg_params
    args.run_dir = run_dir
    if "relative_checkpoint_dir" in cfg_params:
        args.relative_checkpoint_dir = cfg_params['relative_checkpoint_dir']

    print('Launching processes...')
    subprocess_fn(args)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_model_parallel_size', type = int,     default = 1,                                            help = 'tensor_model_parallel_size')
    parser.add_argument('--seed',           type = int,     default = 0,                                            help = 'seed')
    parser.add_argument('--cuda',           type = int,     default = 0,                                            help = 'cuda id')
    parser.add_argument('--world_size',     type = int,     default = 1,                                            help = 'Number of progress')
    parser.add_argument('--per_cpus',       type = int,     default = 1,                                            help = 'Number of perCPUs to use')
    parser.add_argument('--batch_size',     type = int,     default = 32,                                           help = "batch size")
    # parser.add_argument('--predict_len',    type = int,     default = 15,                                           help = "predict len")
    parser.add_argument('--steps',     nargs='+',    type = int,                                           help = "array of sample steps")
    parser.add_argument('--length',         type = int,     default = 16,                                           help = "predict len")
    parser.add_argument('--metric_list',    nargs = '+',                                                            help = 'metric list')
    parser.add_argument('--num_workers',         type = int,     default = 8,                                           help = "worker num")
    # parser.add_argument('--world_size',     type = int,     default = -1,                                           help = 'number of distributed processes')
    parser.add_argument('--init_method',    type = str,     default = 'tcp://127.0.0.1:23456',                      help = 'multi process init method')
    parser.add_argument('--cfgdir',         type = str,     default = '/mnt/petrelfs/chenkang/code/wpredict/IF_WKP/mask_wp/world_size4-16_swin_predictmask_beginnorm_init',  help = 'Where to save the results')
    parser.add_argument('--start_timestamp',    type = str,     default = None,                      help = 'start_timestamp for test')
    parser.add_argument('--end_timestamp',    type = str,     default = None,                      help = 'end_timestamp for test')
    parser.add_argument('--test_save_steps',       type = int,                                           help = "how many steps of predictions should be reserved")
    # debug mode for quick debug #
    parser.add_argument('--debug', action='store_true', help='debug or not')

    args = parser.parse_args()

    main(args)




    # metric list从args中获取，获取meanstd并传到valid中，额外参数怎么设置, 完善metric,是否使用常量数据的操作