import argparse
import os
import torch
from utils.builder import ConfigBuilder
import utils.misc as utils
import yaml
from utils.logger import get_logger
from megatron_utils.tensor_parallel.data import get_data_loader_length


#----------------------------------------------------------------------------

def subprocess_fn(args):
    utils.setup_seed(args.seed * args.world_size + args.rank)

    logger = get_logger("train", args.run_dir, utils.get_rank(), filename='iter.log', resume=args.resume)

    # args.logger = logger
    args.cfg_params["logger"] = logger

    # build config
    logger.info('Building config ...')
    builder = ConfigBuilder(**args.cfg_params)

    logger.info('Building dataloaders ...')

    train_dataloader = builder.get_dataloader(split = 'train')
    logger.info('Train dataloaders build complete')
    valid_dataloader = builder.get_dataloader(split = 'valid')
    logger.info('Valid dataloaders build complete')
    test_dataloader = builder.get_dataloader(split = 'test')
    logger.info('Test dataloaders build complete')
    
    ## set lr_scheduler (by epoch/step) ##
    model_params = args.cfg_params['model']['params']
    ## by step ##
    if "sampler" in args.cfg_params.keys() and "TrainingSampler" in args.cfg_params["sampler"]["type"]:
         lr_scheduler_params = model_params['lr_scheduler']
         for key in lr_scheduler_params:
             for key1 in lr_scheduler_params[key]:
                        if "epochs" in key1:
                            lr_scheduler_params[key][key1] = int(builder.get_max_step() * lr_scheduler_params[key][key1])
    else:
        ## by epoch ##
        steps_per_epoch = get_data_loader_length(train_dataloader)
        if 'lr_scheduler' in model_params:
            lr_scheduler_params = model_params['lr_scheduler']
            for key in lr_scheduler_params:
                if 'by_step' in lr_scheduler_params[key]:
                    if lr_scheduler_params[key]['by_step']:
                        for key1 in lr_scheduler_params[key]:
                            if "epochs" in key1:
                                lr_scheduler_params[key][key1] *= steps_per_epoch
    
    

    # build model
    logger.info('Building models ...')
    model = builder.get_model()
    
    if model.use_ceph:
        model_checkpoint = os.path.join(args.relative_checkpoint_dir, 'checkpoint_latest.pth')
    else:
        model_checkpoint = os.path.join(args.run_dir, 'checkpoint_latest.pth')
    if args.resume:
        # model_checkpoint = 'pafno1f_posMapMul/world_size8-ps4-posMapMul/checkpoint_best.pth'
        resume_checkpoint = args.resume_checkpoint
        model_checkpoint = resume_checkpoint
        ### TODO: important: FOR DEBUG ###
        logger.info("warning: continue finetuning do not load scheduler!!!!!!!!!!!!!!!!!")
        model.load_checkpoint(model_checkpoint, resume=True, load_scheduler=False)

    model_without_ddp = utils.DistributedParallel_Model(model, args.local_rank)

    if args.world_size > 1:
        for key in model_without_ddp.model:
            utils.check_ddp_consistency(model_without_ddp.model[key])

    for key in model_without_ddp.model:
        params = [p for p in model_without_ddp.model[key].parameters() if p.requires_grad]
        cnt_params = sum([p.numel() for p in params])
        logger.info("params {key}: {cnt_params}".format(key=key, cnt_params=cnt_params))



    # valid_dataloader = builder.get_dataloader(split = 'valid')
    # logger.info('valid dataloaders build complete')
    logger.info('begin preprocessing ...')

    # model_without_ddp.stat()
    model_without_ddp.trainer(train_data_loader=train_dataloader, valid_data_loader=valid_dataloader, test_data_loader=test_dataloader,
                               max_epoches=builder.get_max_epoch(), max_steps=builder.get_max_step(), checkpoint_savedir=args.relative_checkpoint_dir if model_without_ddp.use_ceph else args.run_dir, resume=args.resume)

    
def main(args):
    if args.world_size > 1:
        utils.init_distributed_mode(args)
    else:
        args.rank = 0
        args.distributed = False
        # args.local_rank = 0
        torch.cuda.set_device(args.local_rank)
    desc = f'world_size{args.world_size:d}'

    if args.desc is not None:
        desc += f'-{args.desc}'
    
    ##############################################
    # ## trace net output unused ##
    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = "INFO"
    ## or trace unused params in this way ##
    # for name, param in self.model[list(self.model.keys())[0]].named_parameters():
    #         if param.grad is None:
    #             print(name)
    ###############################################
    alg_dir = args.cfg.split("/")[-1].split(".")[0]
    ## check if outdir exists ##
    if not os.path.exists(args.outdir):
        raise ValueError(f"outdir {args.outdir} not exists, should link a oss path")
    args.outdir = args.outdir + "/" + alg_dir
    run_dir = os.path.join(args.outdir, f'{desc}')
    relative_checkpoint_dir = alg_dir + "/" + f'{desc}'
    args.relative_checkpoint_dir = relative_checkpoint_dir
    print(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    train_config_file = os.path.join(run_dir, 'training_options.yaml')

    if not args.resume:
        print("load yaml from config")
        with open(args.cfg, 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    else:
        print("load yaml from resume")
        train_config_file = args.resume_cfg_file
        with open(train_config_file, 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
        del_keys = []
        for key in cfg_params:
            if key in args:
                del_keys.append(key)
        for key in del_keys:
            del cfg_params[key]
    
    # cfg_params['dataloader']['num_workers'] = args.per_cpus

    if args.rank == 0:
        with open(os.path.join(run_dir, 'training_options.yaml'), 'wt') as f:
            yaml.dump(vars(args), f, indent=2, sort_keys=False)
            yaml.dump(cfg_params, f, indent=2, sort_keys=False)

    args.cfg_params = cfg_params
    args.run_dir = run_dir

    args.cfg_params['model']['params']['run_dir']=args.run_dir

    if args.debug:
        args.cfg_params['model']['params']['debug'] = True
    
    if args.visual_vars is not None:
        args.cfg_params['model']['params']['visual_vars'] = args.visual_vars
        args.cfg_params['model']['params']['run_dir'] = args.run_dir
    
    if 'sampler' in args.cfg_params.keys():
        args.cfg_params['model']['params']['sampler_type'] = args.cfg_params['sampler']['type']

    print('Launching processes...')
    subprocess_fn(args)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_model_parallel_size', type = int,     default = 1,                                            help = 'tensor_model_parallel_size')
    parser.add_argument('--resume',                     action = "store_true",                                                  help = 'resume')
    parser.add_argument('--resume_from_config',         action = "store_true",                                                  help = 'resume from config')
    parser.add_argument('--seed',                       type = int,     default = 0,                                            help = 'seed')
    parser.add_argument('--cuda',                       type = int,     default = 0,                                            help = 'cuda id')
    parser.add_argument('--world_size',                 type = int,     default = 1,                                            help = 'Number of progress')
    parser.add_argument('--per_cpus',                   type = int,     default = 1,                                            help = 'Number of perCPUs to use')
    parser.add_argument('--local_rank',                 type=int,       default=0,                                              help='local rank')
    # parser.add_argument('--world_size',     type = int,     default = -1,                                           help = 'number of distributed processes')
    parser.add_argument('--init_method',                type = str,     default='tcp://127.0.0.1:23456',                        help = 'multi process init method')
    parser.add_argument('--outdir',                     type = str,     default='/mnt/petrelfs/chenkang/code/wpredict/IF_WKP',  help = 'Where to save the results')
    parser.add_argument('--cfg', '-c',                  type = str,     default = os.path.join('configs', 'default.yaml'),      help = 'path to the configuration file')
    parser.add_argument('--desc',                       type=str,       default='STR',                                          help = 'String to include in result dir name')
    parser.add_argument('--visual_vars',                nargs='+',       default=None,                                          help = 'visual vars')
    # debug mode for quick debug #
    parser.add_argument('--debug', action='store_true', help='debug or not')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='resume checkpoint')
    parser.add_argument('--resume_cfg_file', type=str, default=None, help='resume from cfg file')

    args = parser.parse_args()

    main(args)