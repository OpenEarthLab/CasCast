if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/cache/gongjunchao/workdir/radar_forecasting')
import torch
import numpy as np
from typing import  Optional, Sequence
from torchmetrics import Metric
from utils.misc import is_dist_avail_and_initialized
import torch.distributed as dist
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import einops
import torch.nn.functional as F
from einops import rearrange
import scipy
from torchvision.transforms.functional import center_crop



@torch.no_grad()
def cal_SSIM(gt, pred, is_img=True):
    '''
    iter_cal=True, gt.shape=pred.shape=[nb b t c h w]
    iter_cal=Fasle, gt.shape=pred.shape=[n t c h w]
    '''
    cal_ssim = StructuralSimilarityIndexMeasure(data_range=int(torch.max(gt)-torch.min(gt)) ).to(gt.device)
    if is_img:
        pred = torch.maximum(pred, torch.min(gt))
        pred = torch.minimum(pred, torch.max(gt))
    pred = einops.rearrange(pred, 'n t c h w -> (n t) c h w')
    gt = einops.rearrange(gt, 'n t c h w -> (n t) c h w')
    ssim = cal_ssim(pred, gt).cpu()
    
    # print(ssim)
    # ssim = cal_ssim_2(pred, gt).cpu()
    
    return ssim.item()

@torch.no_grad()
def cal_PSNR(gt, pred, is_img=True):
    '''
    gt.shape=pred.shape=[n t c h w]
    '''
    cal_psnr = PeakSignalNoiseRatio().to(gt.device)
    if is_img:
        pred = torch.maximum(pred, torch.min(gt))
        pred = torch.minimum(pred, torch.max(gt))
    pred = einops.rearrange(pred, 'n t c h w -> (n t) c h w')
    gt = einops.rearrange(gt, 'n t c h w -> (n t) c h w')
    psnr = 0
    for n in range(pred.shape[0]):
        psnr += cal_psnr(pred[n], gt[n]).cpu()
    return (psnr / pred.shape[0]).item()

@torch.no_grad()
def cal_CRPS(gt, pred, type='avg', scale=4, mode='mean', eps=1e-10):
    """
    gt: (b, t, c, h, w)
    pred: (b, n, t, c, h, w)
    """
    assert mode in ['mean', 'raw'], 'CRPS mode should be mean or raw'
    _normal_dist = torch.distributions.Normal(0, 1)
    _frac_sqrt_pi = 1 / np.sqrt(np.pi)

    b, n, t, _, _, _ = pred.shape
    gt = rearrange(gt, 'b t c h w -> (b t) c h w')
    pred = rearrange(pred, 'b n t c h w -> (b n t) c h w')
    if type == 'avg':
        pred = F.avg_pool2d(pred, scale, stride=scale)
        gt = F.avg_pool2d(gt, scale, stride=scale)
    elif type == 'max':
        pred = F.max_pool2d(pred, scale, stride=scale)
        gt = F.max_pool2d(gt, scale, stride=scale)
    else:
        gt = gt
        pred = pred
    gt = rearrange(gt, '(b t) c h w -> b t c h w', b=b)
    pred = rearrange(pred, '(b n t) c h w -> b n t c h w', b=b, n=n)

    pred_mean = torch.mean(pred, dim=1)
    pred_std = torch.std(pred, dim=1) if n > 1 else torch.zeros_like(pred_mean)
    normed_diff = (pred_mean - gt + eps) / (pred_std + eps)
    cdf = _normal_dist.cdf(normed_diff)
    pdf = _normal_dist.log_prob(normed_diff).exp()

    crps = (pred_std + eps) * (normed_diff * (2 * cdf - 1) + 2 * pdf - _frac_sqrt_pi)
    if mode == "mean":
        return torch.mean(crps).item()
    return crps.item()
    


def _threshold(target, pred ,T):
    t = (target >= T).float()
    p = (pred >= T).float()
    is_nan = torch.logical_or(torch.isnan(target),
                              torch.isnan(pred))
    t[is_nan] = 0
    p[is_nan] = 0
    return t, p



class SEVIRSkillScore(object):
    def __init__(self,
                 layout='NHWT',
                 mode='0',
                 seq_len=None,
                 preprocess_type='sevir',
                 threshold_list=[16, 74, 133, 160, 181, 219],
                 metrics_list=['csi', 'csi-4-avg', 'csi-16-avg',
                               'csi-4-max', 'csi-16-max', 'bias',
                                'sucr', 'pod', 'hss'], #['csi', 'bias', 'sucr', 'pod'],
                 dist_eval=False,
                #  device='cuda',
                 eps=1e-4,):
        self.layout = layout
        self.preprocess_type = preprocess_type
        self.threshold_list = threshold_list
        self.metrics_list = metrics_list
        self.eps = eps
        self.mode = mode
        self.seq_len = seq_len
        
        self.dist_eval = dist_eval
        # self.device = device
        
        if mode in ("0", ):
            self.keep_seq_len_dim = False
            state_shape = (len(self.threshold_list), )
        elif mode in ("1", "2"):
            self.keep_seq_len_dim = True
            assert isinstance(self.seq_len, int), "seq_len must be provided when we need to keep seq_len dim."
            state_shape = (len(self.threshold_list), self.seq_len)
        else:
            raise NotImplementedError(f"mode {mode} not supported!")
        
        self.hits = torch.zeros(state_shape)
        self.misses = torch.zeros(state_shape)
        self.fas = torch.zeros(state_shape)
        self.cor = torch.zeros(state_shape)

        ## pooling csi ##
        self.hits_avg_pool_4 = torch.zeros(state_shape)
        self.misses_avg_pool_4 = torch.zeros(state_shape)
        self.fas_avg_pool_4 = torch.zeros(state_shape)

        self.hits_max_pool_4 = torch.zeros(state_shape)
        self.misses_max_pool_4 = torch.zeros(state_shape)
        self.fas_max_pool_4 = torch.zeros(state_shape)

        self.hits_avg_pool_16 = torch.zeros(state_shape)
        self.misses_avg_pool_16 = torch.zeros(state_shape)
        self.fas_avg_pool_16 = torch.zeros(state_shape)

        self.hits_max_pool_16 = torch.zeros(state_shape)
        self.misses_max_pool_16 = torch.zeros(state_shape)
        self.fas_max_pool_16 = torch.zeros(state_shape)



    def pod(self, hits, misses, fas, eps):
        return hits / (hits + misses + eps)

    def sucr(self, hits, misses, fas, eps):
        return hits / (hits + fas + eps)

    def csi(self, hits, misses, fas, eps):
        return hits / (hits + misses + fas + eps)

    def bias(self, hits, misses, fas, eps):
        bias = (hits + fas) / (hits + misses + eps)
        logbias = torch.pow(bias / torch.log(torch.tensor(2.0)), 2.0)
        return logbias
    
    def hss(self, hits, misses, fas, cor, eps):
        hss = 2 * (hits * cor - misses * fas) / ((hits + misses) * (misses + cor) + (hits + fas) * (fas + cor) + eps)
        return hss


    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        dist.barrier()
        dist.all_reduce(self.hits)
        dist.all_reduce(self.misses)
        dist.all_reduce(self.fas)
        ### avg 4 ###
        dist.all_reduce(self.hits_avg_pool_4)
        dist.all_reduce(self.misses_avg_pool_4)
        dist.all_reduce(self.fas_avg_pool_4)
        ### max 4 ###
        dist.all_reduce(self.hits_max_pool_4)
        dist.all_reduce(self.misses_max_pool_4)
        dist.all_reduce(self.fas_max_pool_4)
        ### avg 16 ###
        dist.all_reduce(self.hits_avg_pool_16)
        dist.all_reduce(self.misses_avg_pool_16)
        dist.all_reduce(self.fas_avg_pool_16)
        ### max 16 ###
        dist.all_reduce(self.hits_max_pool_16)
        dist.all_reduce(self.misses_max_pool_16)
        dist.all_reduce(self.fas_max_pool_16)

    @property
    def hits_misses_fas_reduce_dims(self):
        if not hasattr(self, "_hits_misses_fas_reduce_dims"):
            seq_dim = self.layout.find('T')
            self._hits_misses_fas_reduce_dims = list(range(len(self.layout)))
            if self.keep_seq_len_dim:
                self._hits_misses_fas_reduce_dims.pop(seq_dim)
        return self._hits_misses_fas_reduce_dims

    def preprocess(self, pred, target):
        if self.preprocess_type == "sevir":
            pred = pred.detach() / (1. / 255.)
            target = target.detach() / (1. / 255.)
        elif self.preprocess_type == "meteonet":
            pred = pred.detach() / (1. / 70.)
            target = target.detach() / (1. / 70.)
        else:
            raise NotImplementedError
        return pred, target

    def preprocess_pool(self, pred, target, pool_size=4, type='avg'):
        if self.preprocess_type == "sevir":
            pred = pred.detach() / (1. / 255.)
            target = target.detach() / (1. / 255.)
        elif self.preprocess_type == "meteonet":
            pred = pred.detach() / (1. / 70.)
            target = target.detach() / (1. / 70.)
        b, t, _, _, _ = pred.shape
        pred = rearrange(pred, 'b t c h w -> (b t) c h w')
        target = rearrange(target, 'b t c h w -> (b t) c h w')
        if type == 'avg':
            pred = F.avg_pool2d(pred, kernel_size=pool_size, stride=pool_size)
            target = F.avg_pool2d(target, kernel_size=pool_size, stride=pool_size)
        elif type == 'max':
            pred = F.max_pool2d(pred, kernel_size=pool_size, stride=pool_size)
            target = F.max_pool2d(target, kernel_size=pool_size, stride=pool_size)
        pred = rearrange(pred, '(b t) c h w -> b t c h w', b=b)
        target = rearrange(target, '(b t) c h w -> b t c h w', b=b)
        return pred, target


    def calc_seq_hits_misses_fas(self, pred, target, threshold):
        with torch.no_grad():
            t, p = _threshold(target, pred, threshold)
            hits = torch.sum(t * p, dim=self.hits_misses_fas_reduce_dims).int()
            misses = torch.sum(t * (1 - p), dim=self.hits_misses_fas_reduce_dims).int()
            fas = torch.sum((1 - t) * p, dim=self.hits_misses_fas_reduce_dims).int()
            cor = torch.sum((1 - t) * (1 - p), dim=self.hits_misses_fas_reduce_dims).int()
        return hits, misses, fas, cor
    
    @torch.no_grad()
    def update(self, pred, target):
        ## pool 1 ##
        self.hits = self.hits.to(pred.device)
        self.misses = self.misses.to(pred.device)
        self.fas = self.fas.to(pred.device)
        self.cor = self.cor.to(pred.device)
        _pred, _target = self.preprocess(pred, target)
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits[i] += hits
            self.misses[i] += misses
            self.fas[i] += fas
            self.cor[i] += cor
        ## max pool 4 ##
        self.hits_max_pool_4 = self.hits_max_pool_4.to(pred.device)
        self.misses_max_pool_4 = self.misses_max_pool_4.to(pred.device)
        self.fas_max_pool_4 = self.fas_max_pool_4.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='max')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_max_pool_4[i] += hits
            self.misses_max_pool_4[i] += misses
            self.fas_max_pool_4[i] += fas 
        ## max pool 16 ##
        self.hits_max_pool_16 = self.hits_max_pool_16.to(pred.device)
        self.misses_max_pool_16 = self.misses_max_pool_16.to(pred.device)
        self.fas_max_pool_16 = self.fas_max_pool_16.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='max')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_max_pool_16[i] += hits
            self.misses_max_pool_16[i] += misses
            self.fas_max_pool_16[i] += fas 
        ## avg pool 4 ##
        self.hits_avg_pool_4 = self.hits_avg_pool_4.to(pred.device)
        self.misses_avg_pool_4 = self.misses_avg_pool_4.to(pred.device)
        self.fas_avg_pool_4 = self.fas_avg_pool_4.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='avg')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_avg_pool_4[i] += hits
            self.misses_avg_pool_4[i] += misses
            self.fas_avg_pool_4[i] += fas 
        ## avg pool 16 ##
        self.hits_avg_pool_16 = self.hits_avg_pool_16.to(pred.device)
        self.misses_avg_pool_16 = self.misses_avg_pool_16.to(pred.device)
        self.fas_avg_pool_16 = self.fas_avg_pool_16.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='avg')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_avg_pool_16[i] += hits
            self.misses_avg_pool_16[i] += misses
            self.fas_avg_pool_16[i] += fas 

    def _get_hits_misses_fas(self, metric_name):
        if metric_name.endswith('-4-avg'):
            hits = self.hits_avg_pool_4
            misses = self.misses_avg_pool_4
            fas = self.fas_avg_pool_4
        elif metric_name.endswith('-16-avg'):
            hits = self.hits_avg_pool_16
            misses = self.misses_avg_pool_16
            fas = self.fas_avg_pool_16
        elif metric_name.endswith('-4-max'):
            hits = self.hits_max_pool_4
            misses = self.misses_max_pool_4
            fas = self.fas_max_pool_4
        elif metric_name.endswith('-16-max'):
            hits = self.hits_max_pool_16
            misses = self.misses_max_pool_16
            fas = self.fas_max_pool_16
        else:
            hits = self.hits
            misses = self.misses
            fas = self.fas
        return [hits, misses, fas]
    
    def _get_correct_negtives(self):
        return self.cor
    
    @torch.no_grad()
    def compute(self):
        if self.dist_eval:
            self.synchronize_between_processes()
        
        metrics_dict = {'pod': self.pod,
                        'csi': self.csi,
                        'csi-4-avg': self.csi, 
                        'csi-16-avg': self.csi,
                        'csi-4-max': self.csi, 
                        'csi-16-max': self.csi,
                        'sucr': self.sucr,
                        'bias': self.bias,
                        'hss': self.hss}
        ret = {}
        for threshold in self.threshold_list:
            ret[threshold] = {}
        ret["avg"] = {}
        
        for metrics in self.metrics_list:
            if self.keep_seq_len_dim:
                score_avg = np.zeros((self.seq_len, ))
            else:
                score_avg = 0
            hits, misses, fas = self._get_hits_misses_fas(metrics)
            # scores = metrics_dict[metrics](self.hits, self.misses, self.fas, self.eps)
            if metrics != 'hss':
                scores = metrics_dict[metrics](hits, misses, fas, self.eps)
            else:
                cor = self._get_correct_negtives()
                scores = metrics_dict[metrics](hits, misses, fas, cor, self.eps)
            scores = scores.detach().cpu().numpy()
            for i, threshold in enumerate(self.threshold_list):
                if self.keep_seq_len_dim:
                    score = scores[i]  # shape = (seq_len, )
                else:
                    score = scores[i].item()  # shape = (1, )
                if self.mode in ("0", "1"):
                    ret[threshold][metrics] = score
                elif self.mode in ("2", ):
                    ret[threshold][metrics] = np.mean(score).item()
                else:
                    raise NotImplementedError
                score_avg += score
            score_avg /= len(self.threshold_list)
            if self.mode in ("0", "1"):
                ret["avg"][metrics] = score_avg
            elif self.mode in ("2",):
                ret["avg"][metrics] = np.mean(score_avg).item()
            else:
                raise NotImplementedError
        return ret

    @torch.no_grad()
    def get_single_frame_metrics(self, target, pred, metrics=['ssim', 'psnr', ]): #'cspr', 'cspr-4-avg', 'cspr-16-avg', 'cspr-4-max', 'cspr-16-max'
        metric_funcs = {
            'ssim': cal_SSIM,
            'psnr': cal_PSNR
        }
        metrics_dict = {}
        for metric in metrics:
            metric_fun = metric_funcs[metric]
            metrics_dict[metric] = metric_fun(gt=target*255., pred=pred*255., is_img=False)
        return metrics_dict
    
    @torch.no_grad()
    def get_crps(self, target, pred):
        """
        pred: (b, t, c, h, w)/(b, n, t, c, h, w)
        target: (b, t, c, h, w)
        """
        if len(pred.shape) == 5:
            pred = pred.unsqueeze(1)
        crps = cal_CRPS(gt=target, pred=pred, type='none')
        crps_avg_4 = cal_CRPS(gt=target, pred=pred, type='avg', scale=4)
        crps_avg_16 = cal_CRPS(gt=target, pred=pred, type='avg', scale=16)
        crps_max_4 = cal_CRPS(gt=target, pred=pred, type='max', scale=4)
        crps_max_16 = cal_CRPS(gt=target, pred=pred, type='max', scale=16)
        crps_dict = {
            'crps': crps,
            'crps_avg_4': crps_avg_4,
            'crps_avg_16': crps_avg_16,
            'crps_max_4': crps_max_4,
            'crps_max_16': crps_max_16
        }
        return crps_dict



    def reset(self):
        self.hits = self.hits*0
        self.misses = self.misses*0
        self.fas = self.fas*0

        self.hits_avg_pool_4 *= 0
        self.hits_avg_pool_16 *= 0
        self.hits_max_pool_4 *= 0
        self.hits_max_pool_16 *= 0

        self.misses_avg_pool_4 *= 0
        self.misses_avg_pool_16 *= 0
        self.misses_max_pool_4 *= 0
        self.misses_max_pool_16 *= 0
 
        self.fas_avg_pool_4 *= 0
        self.fas_avg_pool_16 *= 0
        self.fas_max_pool_4  *= 0
        self.fas_max_pool_16  *= 0


@torch.no_grad()
class cal_FVD:
    def __init__(self, use_gpu=False, resize_crop=False):
        '''
        iter_cal=True, gt.shape=pred.shape=[nb b t c h w]
        iter_cal=Fasle, gt.shape=pred.shape=[n t c h w]
        '''
        
        self.use_gpu = use_gpu
        self.resize_crop = resize_crop
        # detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
        self.detector = torch.jit.load("/mnt/cache/gongjunchao/workdir/radar_forecasting/utils/fvd/i3d_torchscript.pt").eval()
        if torch.cuda.is_available() and self.use_gpu:
            self.detector = self.detector.cuda()
        self.feats = []
    
    def preprocess(self, video):
        """
        video: (b, t, c, h, w) in [0, 1]
        this function transform the domain to [-1, 1] 
        """
        video = video * 2 - 1
        return video

    @torch.no_grad()
    def __call__(self, videos_real, videos_fake):
        feats_fake = []
        feats_real = []
        detector_kwargs = dict(rescale=False, resize=False, return_features=True)
        
        videos_fake = einops.rearrange(
            self.bilinear_interpolation(videos_fake), 'n t c h w -> n c t h w'
        )
        videos_real = einops.rearrange(
            self.bilinear_interpolation(videos_real), 'n t c h w -> n c t h w'
        )
        if torch.cuda.is_available() and self.use_gpu:
            videos_fake, videos_real = videos_fake.cuda(), videos_real.cuda()
        # print(videos_fake.shape, videos_real.shape)
        # videos_fake = videos_fake.repeat(1, 1, 10, 1, 1)
        # videos_real = videos_real.repeat(1, 1, 10, 1, 1)
        feats_fake = self.detector(videos_fake, **detector_kwargs).cpu()
        feats_real = self.detector(videos_real, **detector_kwargs).cpu()
        self.feats.append(torch.stack([feats_fake, feats_real], dim=0))
        return
    
    def update(self, videos_real, videos_fake):
        self(videos_real=videos_real, videos_fake=videos_fake)
        return

    def _reset(self):
        self.feats = []

    def compute(self):
        feats = torch.cat(self.feats, dim=1)
        fake_feats = feats[0]
        real_feats = feats[1]
        fvd = self._cal_FVD(feats_fake=fake_feats, feats_real=real_feats)
        return fvd

    def bilinear_interpolation(self, image):
        N, T, C, H, W = image.shape
        def my_resize(img):
            img = img.view(-1, C, H, W)
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            img = img.view(N, T, C, 224, 224)  
            return img
        def my_resize_crop(img):
            img = img.view(-1, C, H, W)
            if H<W:
                img = F.interpolate(img, size=(224, int(W*224/H)), mode='bilinear', align_corners=False)
                img = img.view(N, T, C, 224, int(W*224/H))  
            else:   # W<=H
                img = F.interpolate(img, size=(int(H*224/W), 224), mode='bilinear', align_corners=False)
                img = img.view(N, T, C, int(H*224/W), 224)  
            return center_crop(img, (224, 224))
        if H == W and H < 224:
            return my_resize(img=image)
        elif self.resize_crop:
            return my_resize_crop(img=image)
        else: 
            return my_resize(img=image)

    def _cal_FVD(self, feats_fake, feats_real):
        def compute_fvd(feats_fake, feats_real):
            mu_gen, sigma_gen = compute_stats(feats_fake)
            mu_real, sigma_real = compute_stats(feats_real)
            m = np.square(mu_gen - mu_real).sum()
            s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
            fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
            return float(fid)

        def compute_stats(feats):
            feats = feats.reshape(-1, feats.shape[-1])
            mu = feats.mean(axis=0)
            sigma = np.cov(feats, rowvar=False)
            return mu, sigma
        return compute_fvd(feats_fake, feats_real)
    