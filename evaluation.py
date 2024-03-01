import argparse
import os
import torch
from utils.builder import ConfigBuilder
import utils.misc as utils
import yaml
from utils.logger import get_logger
import copy
from utils.misc import is_dist_avail_and_initialized



#----------------------------------------------------------------------------

def subprocess_fn(args):
    utils.setup_seed(args.seed * args.world_size + args.rank)

    logger = get_logger("test", args.run_dir, utils.get_rank(), filename=f'{args.test_name}.log')

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
    
    ### evaluation metric evaluator ###
    if args.metrics_type == 'hko7':
        raise NotImplementedError
    elif args.metrics_type == 'hko7_official':
        from utils.metrics import HKOEvaluation_official
        model_without_ddp.eval_metrics = HKOEvaluation_official(layout='NTCHW', seq_len=args.pred_len, dist_eval=True if is_dist_avail_and_initialized() else False)
        model_without_ddp.metrics_type = 'hko7_official'
    elif args.metrics_type == 'SEVIRSkillScore':
        from utils.metrics import SEVIRSkillScore
        model_without_ddp.eval_metrics = SEVIRSkillScore(layout='NTCHW', seq_len=args.pred_len, mode='1',
                                                         dist_eval=True if is_dist_avail_and_initialized() else False)
        model_without_ddp.metrics_type = 'SEVIRSkillScore'
    elif args.metrics_type == 'METEONETScore':
        from utils.metrics import SEVIRSkillScore
        model_without_ddp.eval_metrics = SEVIRSkillScore(layout='NTCHW', seq_len=args.pred_len, 
                                                         threshold_list=[19, 28, 35, 40, 47], preprocess_type='meteonet', 
                                                dist_eval=True if is_dist_avail_and_initialized() else False)
        model_without_ddp.metrics_type = 'METEONETScore'
    else:
        raise NotImplementedError
    
    ### evaluation visualizer ####
    if model_without_ddp.visualizer_type == 'sevir_visualizer':
        from utils.visualizer import sevir_visualizer
        model_without_ddp.visualizer = sevir_visualizer(exp_dir=args.run_dir, sub_dir=f'{args.test_name}_vis')
    elif model_without_ddp.visualizer_type == 'hko7_visualizer':
        from utils.visualizer import hko7_visualizer
        model_without_ddp.visualizer = hko7_visualizer(exp_dir=args.run_dir, sub_dir=f'{args.test_name}_vis')
    elif model_without_ddp.visualizer_type == 'meteonet_visualizer':
            from utils.visualizer import meteonet_visualizer
            model_without_ddp.visualizer = meteonet_visualizer(exp_dir=args.run_dir, sub_dir=f'{args.test_name}_vis')
    else:
        raise NotImplementedError
    # if args.test_save_steps > 0 :
    #     model_without_ddp.checkpoint_savedir = os.path.join(args.relative_checkpoint_dir, args.start_timestamp)
    model_without_ddp.test_final(test_dataloader, args.pred_len)

def main(args):
    if args.world_size > 1:
        utils.init_distributed_mode(args)
    else:
        args.rank = 0
        # args.local_rank = 0
        args.distributed = False
        args.gpu = 0
        torch.cuda.set_device(args.gpu)


    run_dir = args.cfgdir
    print(run_dir)
    
    # args.cfg = os.path.join(args.cfgdir, 'lora_eval_options.yaml')
    args.cfg = os.path.join(args.cfgdir, 'training_options.yaml')
    with open(args.cfg, 'r') as cfg_file:
        cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    # 根据申请的cpu数来设置dataloader的线程数
    cfg_params['dataloader']['num_workers'] = args.num_workers
    cfg_params['dataset']['test'] = copy.deepcopy(cfg_params['dataset']['valid'])
    cfg_params['dataset']['test']['pred_length'] = args.pred_len

    if "checkpoint_path" in cfg_params["model"]["params"]["extra_params"]:
        del cfg_params["model"]["params"]["extra_params"]["checkpoint_path"]

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
    parser.add_argument('--local_rank',                 type=int,       default=0,                                              help='local rank')
    # parser.add_argument('--predict_len',    type = int,     default = 15,                                           help = "predict len")
    parser.add_argument('--num_workers',         type = int,     default = 8,                                           help = "worker num")
    # parser.add_argument('--world_size',     type = int,     default = -1,                                           help = 'number of distributed processes')
    parser.add_argument('--init_method',    type = str,     default = 'tcp://127.0.0.1:23456',                      help = 'multi process init method')
    parser.add_argument('--cfgdir',         type = str,     default = '/mnt/petrelfs/chenkang/code/wpredict/IF_WKP/mask_wp/world_size4-16_swin_predictmask_beginnorm_init',  help = 'Where to save the results')
    parser.add_argument('--pred_len',       type = int,     default = 10,                                           help = "predict len")
    parser.add_argument('--metrics_type',   type = str,     default='hko7',                                         help = 'evaluator to test model')
    parser.add_argument('--test_name',    type = str,     default='test',                                         help = 'test name')
    # debug mode for quick debug #
    parser.add_argument('--debug', action='store_true', help='debug or not')

    args = parser.parse_args()

    main(args)