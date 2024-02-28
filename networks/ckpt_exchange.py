import torch



if __name__ == "__main__":
    official_weight_path = '/mnt/lustre/gongjunchao/earthformer_sevir.pt'
    official_weight_ckpt = torch.load(official_weight_path)

    template_ckpt_path = '/mnt/lustre/gongjunchao/checkpoint_latest.pth'
    template_ckpt = torch.load(template_ckpt_path)

    sub_key = 'model'
    model_name = 'EarthFormer_xy'
    for k, v in template_ckpt[sub_key][model_name].items():
        template_ckpt[sub_key][model_name][k] = official_weight_ckpt[k[4:]]

    torch.save(template_ckpt, '/mnt/lustre/gongjunchao/official_checkpoint_earthformer_sevir.pth')
    import pdb; pdb.set_trace()

# srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u ckpt_exchange.py #