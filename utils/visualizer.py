if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/cache/gongjunchao/workdir/radar_forecasting')

from utils.misc import get_rank, get_world_size, is_dist_avail_and_initialized
import numpy as np
from megatron_utils import mpu
import matplotlib.pyplot as plt
import os
import io

from datasets.sevir_util.sevir_cmap import get_cmap, VIL_COLORS, VIL_LEVELS
from typing import Optional, Sequence, Union, Dict
import math
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib import colors


try:
    from petrel_client.client import Client
except:
    pass




class non_visualizer(object):
    pass



class meteonet_visualizer(object):
    def __init__(self, exp_dir, sub_dir='meteonet_train_vis'):
        self.exp_dir = exp_dir
        # self.hko_zr_a = 58.53
        # self.hko_zr_b = 1.56
        self.sub_dir = sub_dir
        self.cmap_color = 'gist_ncar'
        self.save_dir = f'{self.exp_dir}/{sub_dir}'
        os.makedirs(exist_ok=True, name=self.save_dir)

        self.client = Client("~/petreloss.conf")
    
    def save_pixel_image(self, pred_image, target_img, step):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            pred_imgs = pred_image.detach().cpu().numpy() ##b, t, c, h, w
            pxl_pred_imgs = pred_imgs[0, :, 0] * 70 # to pixel

            target_imgs = target_img.detach().cpu().numpy()
            pxl_target_imgs = target_imgs[0, :,  0] * 70

            pred_imgs = pxl_pred_imgs 
            target_imgs = pxl_target_imgs

            for t in range(pred_imgs.shape[0]):
                pred_img = pred_imgs[t]
                target_img = target_imgs[t]
                ## plot pred_img and target_img pair
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                val_max = 70
                val_min = 0
                ax1.imshow(pred_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax1.set_title(f'pred_pixel_step{step}_time{t}')
                im2 = ax2.imshow(target_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax2.set_title(f'target_pixel')
                cbar1 = plt.colorbar(im2, ax=[ax2, ax1])

                # mse = np.square(pred_img - target_img)
                # im3 = ax3.imshow(mse, cmap='OrRd')
                # ax3.set_title(f'mse')
                # cbar2 = plt.colorbar(im3, ax=ax3)
                plt.savefig(f'{self.save_dir}/pixel_step{step}_time{t}.png', dpi=300, bbox_inches='tight', pad_inches=0)
                plt.clf()
    
    def save_meteo_last_image_and_npy(self, pred_image, target_img, step, ceph_prefix):
        assert get_world_size() == 1
        pred_imgs = pred_image.detach().cpu().numpy() ##b, t, c, h, w
        pxl_pred_imgs = pred_imgs[0, :, 0] * 70 # to pixel

        target_imgs = target_img.detach().cpu().numpy()
        pxl_target_imgs = target_imgs[0, :,  0] * 70

        ### define color map ###
        cmap = colors.ListedColormap(['lavender','indigo','mediumblue','dodgerblue', 'skyblue','cyan',
                          'olivedrab','lime','greenyellow','orange','red','magenta','pink'])
        # Reflectivity : colorbar definition
        if (np.max(pxl_target_imgs) > 56):
            borne_max = np.max(pxl_target_imgs)
        else:
            borne_max = 56 + 10
        bounds = [0,4,8,12,16,20,24,32,40,48,56,borne_max]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        ## save last frame ##
        last_pred_img = pxl_pred_imgs[-1]
        last_target_img = pxl_target_imgs[-1]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.imshow(last_pred_img, cmap=cmap, norm=norm)
        ax1.set_title(f'pred_meteo_step{step}_60min')
        im2 = ax2.imshow(last_target_img, cmap=cmap, norm=norm)
        ax2.set_title(f'target_meteo_60min')
        cbar1 = plt.colorbar(im2, ax=[ax2, ax1])

        plt.savefig(f'{self.save_dir}/meteo_step{step}_60min.png', dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()

        ## save npy to ceph ##
        with io.BytesIO() as f:
            np.save(f, pxl_pred_imgs)
            f.seek(0)
            self.client.put(f'{ceph_prefix}/pred_step{step}.npy', f)




class sevir_visualizer(object):
    def __init__(self, exp_dir, sub_dir='sevir_train_vis'):
        self.exp_dir = exp_dir
        # self.hko_zr_a = 58.53
        # self.hko_zr_b = 1.56
        self.cmap_color = 'gist_ncar'
        self.sub_dir = sub_dir
        self.save_dir = f'{self.exp_dir}/{sub_dir}'
        os.makedirs(exist_ok=True, name=self.save_dir)

        self.client = Client("~/petreloss.conf")
    
    def save_pixel_image(self, pred_image, target_img, step):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            pred_imgs = pred_image.detach().cpu().numpy() ##b, t, c, h, w
            pxl_pred_imgs = pred_imgs[0, :, 0] * 255 # to pixel

            target_imgs = target_img.detach().cpu().numpy()
            pxl_target_imgs = target_imgs[0, :,  0] * 255

            pred_imgs = pxl_pred_imgs 
            target_imgs = pxl_target_imgs

            for t in range(pred_imgs.shape[0]):
                pred_img = pred_imgs[t]
                target_img = target_imgs[t]
                ## plot pred_img and target_img pair
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                val_max = 255
                val_min = 0
                ax1.imshow(pred_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax1.set_title(f'pred_pixel_step{step}_time{t}')
                im2 = ax2.imshow(target_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax2.set_title(f'target_pixel')
                cbar1 = plt.colorbar(im2, ax=[ax2, ax1])

                # mse = np.square(pred_img - target_img)
                # im3 = ax3.imshow(mse, cmap='OrRd')
                # ax3.set_title(f'mse')
                # cbar2 = plt.colorbar(im3, ax=ax3)
                plt.savefig(f'{self.save_dir}/pixel_step{step}_time{t}.png', dpi=150, bbox_inches='tight', pad_inches=0)
                plt.clf()
        
    def save_npy(self, pred_image, target_img, step):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:    
            pred_imgs = pred_image.detach().cpu().numpy()[:1]
            target_imgs = target_img.detach().cpu().numpy()[:1]

            np.save(f'{self.save_dir}/pred_step{step}.npy', pred_imgs)
            np.save(f'{self.save_dir}/gt_step{step}.npy', target_imgs)

                
    
    def cmap_dict(self, s):
        return {'cmap': get_cmap(s, encoded=True)[0],
                'norm': get_cmap(s, encoded=True)[1],
                'vmin': get_cmap(s, encoded=True)[2],
                'vmax': get_cmap(s, encoded=True)[3]}
    
    def save_vil_image(self, pred_image, target_img, step):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            pred_imgs = pred_image.detach().cpu().numpy() ##b, t, c, h, w
            pxl_pred_imgs = pred_imgs[0, :, 0] * 255 # to pixel

            target_imgs = target_img.detach().cpu().numpy()
            pxl_target_imgs = target_imgs[0, :,  0] * 255

            pred_imgs = pxl_pred_imgs 
            target_imgs = pxl_target_imgs

            for t in range(pred_imgs.shape[0]):
                pred_img = pred_imgs[t]
                target_img = target_imgs[t]
                ## plot pred_img and target_img pair
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                ax1.imshow(pred_img, **self.cmap_dict('vil'))
                ax1.set_title(f'pred_vil_step{step}_time{t}')
                im2 = ax2.imshow(target_img, **self.cmap_dict('vil'))
                ax2.set_title(f'target_vil')
                cbar1 = plt.colorbar(im2, ax=[ax2, ax1])

                # mse = np.square(pred_img - target_img)
                # im3 = ax3.imshow(mse, cmap='OrRd')
                # ax3.set_title(f'mse')
                # cbar2 = plt.colorbar(im3, ax=ax3)
                plt.savefig(f'{self.save_dir}/vil_step{step}_time{t}.png', dpi=300, bbox_inches='tight', pad_inches=0)
                plt.clf()
    
    def save_vil_last_image_and_npy(self, pred_image, target_img, step, ceph_prefix):
        assert get_world_size() == 1
        pred_imgs = pred_image.detach().cpu().numpy() ##b, t, c, h, w
        pxl_pred_imgs = pred_imgs[0, :, 0] * 255 # to pixel

        target_imgs = target_img.detach().cpu().numpy()
        pxl_target_imgs = target_imgs[0, :,  0] * 255

        ## save last frame ##
        last_pred_img = pxl_pred_imgs[-1]
        last_target_img = pxl_target_imgs[-1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.imshow(last_pred_img, **self.cmap_dict('vil'))
        ax1.set_title(f'pred_vil_step{step}_60min')
        im2 = ax2.imshow(last_target_img, **self.cmap_dict('vil'))
        ax2.set_title(f'target_vil_60min')
        cbar1 = plt.colorbar(im2, ax=[ax2, ax1])

        plt.savefig(f'{self.save_dir}/vil_step{step}_60min.png', dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()

        ## save npy to ceph ##
        with io.BytesIO() as f:
            np.save(f, pxl_pred_imgs)
            f.seek(0)
            self.client.put(f'{ceph_prefix}/pred_step{step}.npy', f)
        



class hko7_visualizer(object):
    def __init__(self, exp_dir, sub_dir='train_vis'):
        self.exp_dir = exp_dir
        self.hko_zr_a = 58.53
        self.hko_zr_b = 1.56
        self.sub_dir = sub_dir
        self.cmap_color = 'gist_ncar'
        self.save_dir = f'{self.exp_dir}/hko7_{sub_dir}'
        os.makedirs(exist_ok=True, name=self.save_dir)

        self.client = Client("~/petreloss.conf")
    
    def save_pixel_image(self, pred_image, target_img, step):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            pred_imgs = pred_image.detach().cpu().numpy() ##b, t, c, h, w
            pxl_pred_imgs = pred_imgs[0, :, 0]* 255 # to pixel

            target_imgs = target_img.detach().cpu().numpy()
            pxl_target_imgs = target_imgs[0, :,  0]* 255

            for t in range(pxl_pred_imgs.shape[0]):
                pxl_pred_img = pxl_pred_imgs[t]
                pxl_target_img = pxl_target_imgs[t]
                ## plot pred_img and target_img pair
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                val_max = 255
                val_min = 0
                ax1.imshow(pxl_pred_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax1.set_title(f'pred_pixel_step{step}_time{t}')
                im2 = ax2.imshow(pxl_target_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax2.set_title(f'target_pixel')
                cbar1 = plt.colorbar(im2, ax=[ax2, ax1])

                # mse = np.square(pred_img - target_img)
                # im3 = ax3.imshow(mse, cmap='OrRd')
                # ax3.set_title(f'mse')
                # cbar2 = plt.colorbar(im3, ax=ax3)
                plt.savefig(f'{self.save_dir}/pixel_step{step}_time{t}.png', dpi=300, bbox_inches='tight', pad_inches=0)
                plt.clf()

    def save_dbz_image(self, pred_image, target_img, step):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            pred_imgs = pred_image.detach().cpu().numpy() ##b, t, c, h, w
            pxl_pred_imgs = pred_imgs[0, :, 0] #* 255 # to pixel

            target_imgs = target_img.detach().cpu().numpy()
            pxl_target_imgs = target_imgs[0, :,  0] #* 255

            pred_imgs = self._pixel_to_dBZ(pxl_pred_imgs) #self._pixel_to_rainfall(pxl_pred_imgs)
            target_imgs = self._pixel_to_dBZ(pxl_target_imgs) #self._pixel_to_rainfall(pxl_target_imgs)

            for t in range(pred_imgs.shape[0]):
                pred_img = pred_imgs[t]
                target_img = target_imgs[t]
                ## plot pred_img and target_img pair
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                val_max = np.max(target_img)
                val_min = np.min(target_img)
                ax1.imshow(pred_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax1.set_title(f'pred_dbz_step{step}_time{t}')
                im2 = ax2.imshow(target_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax2.set_title(f'target_dbz')
                cbar1 = plt.colorbar(im2, ax=[ax2, ax1])

                # mse = np.square(pred_img - target_img)
                # im3 = ax3.imshow(mse, cmap='OrRd')
                # ax3.set_title(f'mse')
                # cbar2 = plt.colorbar(im3, ax=ax3)
                plt.savefig(f'{self.save_dir}/dbz_step{step}_time{t}.png', dpi=300, bbox_inches='tight', pad_inches=0)
                plt.clf()
    
    def save_hko7_last_image_and_npy(self, pred_image, target_img, step, ceph_prefix):
        assert get_world_size() == 1
        pred_imgs = pred_image.detach().cpu().numpy() ##b, t, c, h, w
        pxl_pred_imgs = pred_imgs[0, :, 0] * 255 # to pixel

        target_imgs = target_img.detach().cpu().numpy()
        pxl_target_imgs = target_imgs[0, :,  0] * 255

        ## save last frame ##
        last_pred_img = pxl_pred_imgs[-1]
        last_target_img = pxl_target_imgs[-1]

        val_max = 255
        val_min = 0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.imshow(last_pred_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
        ax1.set_title(f'pred_pxl_step{step}_60min')
        im2 = ax2.imshow(last_target_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
        ax2.set_title(f'target_pxl_60min')
        cbar1 = plt.colorbar(im2, ax=[ax2, ax1])

        plt.savefig(f'{self.save_dir}/pxl_step{step}_60min.png', dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()

        ## save npy to ceph ##
        with io.BytesIO() as f:
            np.save(f, pxl_pred_imgs)
            f.seek(0)
            self.client.put(f'{ceph_prefix}/pred_step{step}.npy', f)



    def _pixel_to_rainfall(self, img, a=None, b=None):
        """Convert the pixel values to real rainfall intensity

        Parameters
        ----------
        img : np.ndarray
        a : float32, optional
        b : float32, optional

        Returns
        -------
        rainfall_intensity : np.ndarray
        """
        if a is None:
            a = self.hko_zr_a
        if b is None:
            b = self.hko_zr_b
        dBZ = self._pixel_to_dBZ(img)
        dBR = (dBZ - 10.0 * np.log10(a)) / b
        rainfall_intensity = np.power(10, dBR / 10.0)
        return rainfall_intensity
    
    def _pixel_to_dBZ(self, img):
        """

        Parameters
        ----------
        img : np.ndarray or float

        Returns
        -------

        """
        return img * 70.0 - 10.0


def vis_sevir_seq(
        save_path,
        seq: Union[np.ndarray, Sequence[np.ndarray]],
        label: Union[str, Sequence[str]] = "pred",
        norm: Optional[Dict[str, float]] = None,
        interval_real_time: float = 10.0,  plot_stride=2,
        label_rotation=0,
        label_offset=(-0.06, 0.4),
        label_avg_int=False,
        fs=10,
        max_cols=10, ):
    """
    Parameters
    ----------
    seq:    Union[np.ndarray, Sequence[np.ndarray]]
        shape = (T, H, W). Float value 0-1 after `norm`.
    label:  Union[str, Sequence[str]]
        label for each sequence.
    norm:   Union[str, Dict[str, float]]
        seq_show = seq * norm['scale'] + norm['shift']
    interval_real_time: float
        The minutes of each plot interval
    max_cols: int
        The maximum number of columns in the figure.
    """

    def cmap_dict(s):
        return {'cmap': get_cmap(s, encoded=True)[0],
                'norm': get_cmap(s, encoded=True)[1],
                'vmin': get_cmap(s, encoded=True)[2],
                'vmax': get_cmap(s, encoded=True)[3]}

    # cmap_dict = lambda s: {'cmap': get_cmap(s, encoded=True)[0],
    #                        'norm': get_cmap(s, encoded=True)[1],
    #                        'vmin': get_cmap(s, encoded=True)[2],
    #                        'vmax': get_cmap(s, encoded=True)[3]}

    fontproperties = FontProperties()
    fontproperties.set_family('serif')
    # font.set_name('Times New Roman')
    fontproperties.set_size(fs)
    # font.set_weight("bold")

    if isinstance(seq, Sequence):
        seq_list = [ele.astype(np.float32) for ele in seq]
        assert isinstance(label, Sequence) and len(label) == len(seq)
        label_list = label
    elif isinstance(seq, np.ndarray):
        seq_list = [seq.astype(np.float32), ]
        assert isinstance(label, str)
        label_list = [label, ]
    else:
        raise NotImplementedError
    if label_avg_int:
        label_list = [f"{ele1}\nAvgInt = {np.mean(ele2): .3f}"
                      for ele1, ele2 in zip(label_list, seq_list)]
    # plot_stride
    seq_list = [ele[::plot_stride, ...] for ele in seq_list]
    seq_len_list = [len(ele) for ele in seq_list]

    max_len = max(seq_len_list)

    max_len = min(max_len, max_cols)
    seq_list_wrap = []
    label_list_wrap = []
    seq_len_list_wrap = []
    for i, (seq, label, seq_len) in enumerate(zip(seq_list, label_list, seq_len_list)):
        num_row = math.ceil(seq_len / max_len)
        for j in range(num_row):
            slice_end = min(seq_len, (j + 1) * max_len)
            seq_list_wrap.append(seq[j * max_len: slice_end])
            if j == 0:
                label_list_wrap.append(label)
            else:
                label_list_wrap.append("")
            seq_len_list_wrap.append(min(seq_len - j * max_len, max_len))

    if norm is None:
        norm = {'scale': 255,
                'shift': 0}
    nrows = len(seq_list_wrap)
    fig, ax = plt.subplots(nrows=nrows,
                           ncols=max_len,
                           figsize=(3 * max_len, 3 * nrows))

    for i, (seq, label, seq_len) in enumerate(zip(seq_list_wrap, label_list_wrap, seq_len_list_wrap)):
        ax[i][0].set_ylabel(ylabel=label, fontproperties=fontproperties, rotation=label_rotation)
        ax[i][0].yaxis.set_label_coords(label_offset[0], label_offset[1])
        for j in range(0, max_len):
            if j < seq_len:
                x = seq[j] * norm['scale'] + norm['shift']
                ax[i][j].imshow(x, **cmap_dict('vil'))
                if i == len(seq_list) - 1 and i > 0:  # the last row which is not the `in_seq`.
                    ax[-1][j].set_title(f"Min {int(interval_real_time * (j + 1) * plot_stride)}",
                                        y=-0.25, fontproperties=fontproperties)
            else:
                ax[i][j].axis('off')

    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].xaxis.set_ticks([])
            ax[i][j].yaxis.set_ticks([])

    # Legend of thresholds
    num_thresh_legend = len(VIL_LEVELS) - 1
    legend_elements = [Patch(facecolor=VIL_COLORS[i],
                             label=f'{int(VIL_LEVELS[i - 1])}-{int(VIL_LEVELS[i])}')
                       for i in range(1, num_thresh_legend + 1)]
    ax[0][0].legend(handles=legend_elements, loc='center left',
                    bbox_to_anchor=(-1.2, -0.),
                    borderaxespad=0, frameon=False, fontsize='10')
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(save_path)
    plt.close(fig)

if __name__ == "__main__":
    print("start")
    import sys
    sys.path.append('/mnt/cache/gongjunchao/workdir/radar_forecasting')
    import torch
    from datasets.sevir import get_sevir_dataset
    dataset_kwargs = {
        'split': 'valid',
        'input_length': 13,
        'pred_length': 12,
        'data_dir': 'radar:s3://weather_radar_datasets/sevir'
    }
    # def __init__(self, split, input_length, pred_length, base_freq, height=480, width=480, **kwargs):
    dataset = get_sevir_dataset(**dataset_kwargs)
    # visualizer = hko7_visualizer(exp_dir='.')
    visualizer = sevir_visualizer(exp_dir='.')
    data_dict = dataset.__getitem__(3001)

    inp = data_dict['data_samples']
    visualizer.save_vil_image(inp.unsqueeze(0), inp.unsqueeze(0), 0)
    # vis_sevir_seq('test.png', inp.squeeze(1).numpy(), label='pred', interval_real_time=5.0, plot_stride=1, max_cols=11)
### srun -p ai4earth --kill-on-bad-exit=1 --quotatype=reserved --gres=gpu:0 python -u visualizer.py ###
