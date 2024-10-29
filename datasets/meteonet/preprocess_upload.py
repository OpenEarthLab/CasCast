import numpy as np
import os
import datetime
import copy
from matplotlib import pyplot as plt

from petrel_client.client import Client
import io



client = Client("~/petreloss.conf")
oss_prefix = 'radar:'
upload_bucket = 'meteonet_data'


def date_to_index(date):
    day = date.day
    hour = date.hour
    minute = date.minute

    if day == 31:
        rel_day = 10
    else:
        rel_day = day%10 - 1
        if rel_day < 0:
            rel_day = 9
    
    index_rel_day = rel_day * 24 * 12
    index_hour = hour * 12
    index_minute = minute // 5

    index = index_rel_day + index_hour + index_minute
    return index

def complete_miss_dates_in_data(data, miss_index):
    data_copy = copy.deepcopy(data)
    miss_template = np.zeros((1, data.shape[-2], data.shape[-1]))
    for i in miss_index:
        data_copy = np.insert(data_copy, i, miss_template, axis=0)
    
    ## check data_copy
    checksum = 0
    for i in miss_index:
        checksum += data_copy[i].sum()
    assert checksum == 0
    try:
        assert (data_copy.shape[0]==2880 or data_copy.shape[0]==3168 or data_copy.shape[0]==2592 or data_copy.shape[0]==2304) ##考虑2月 
    except:
        import pdb; pdb.set_trace()
    return data_copy

def split_data_to_continue_frames(comp_data, miss_index, valid_data_num):
    continue_frames = []
    if len(miss_index) > 0:
        for i, miss_ind in enumerate(miss_index):
            if i == 0:
                continue_clip = comp_data[:miss_index[i]]
            else:
                continue_clip = comp_data[miss_index[i-1]+1:miss_index[i]]
            if continue_clip.shape[0] > 0:
                continue_frames.append(continue_clip)
        ## add last clip ##
        try:
            last_clip = comp_data[miss_index[-1]+1:]
        except:
            import pdb; pdb.set_trace()
        if last_clip.shape[0] > 0:
            continue_frames.append(last_clip)
    else:
        continue_frames.append(comp_data)
    ## check continue_frames ##
    total_frame_num = 0
    for clip in continue_frames:
        total_frame_num += clip.shape[0]
    assert total_frame_num == valid_data_num
    return continue_frames

def event_filtering(continue_frames, pixel_threshold, l_in, l_out):
    event_set = []
    i = 20
    while i+l_out < len(continue_frames):
        if continue_frames[i].mean() > pixel_threshold:
            event = continue_frames[i-l_in:i+l_out]
            try:
                assert event.shape[0] == l_in + l_out
            except:
                import pdb; pdb.set_trace()
            event_pixel_value = event.mean()
            if event_pixel_value >= pixel_threshold:
                event_set.append(event)
                i = i + l_out 
        i = i + 1
    return event_set

def upload_event(event, event_id):
    # import pdb; pdb.set_trace()
    save_path = os.path.join(f'{oss_prefix}s3://{upload_bucket}', '24Frames', f'{event_id}.npy')
    with io.BytesIO() as f:
        np.save(f, event)
        f.seek(0)
        client.put(save_path, f.read())
    return 


if __name__ == "__main__":
    ## top left:400x400, in 12 out 12 ##
    total_event = 0
    sub_region = 400
    pixel_threshold = 0.5
    l_in = 12
    l_out = 12

    event_id = 0
    split = 'train'
    print('train split:')
    for y in [2016, 2017]:
        for m in range(1, 13):
            for p in range(1, 4):
                path = os.path.join('/mnt/lustre/gongjunchao/meteonet_eg',
                        f'SE_reflectivity_old_product_{y}', 
                        f'reflectivity-old-SE-{y}-{m:02d}',
                        f'reflectivity_old_SE_{y}_{m:02d}.{p}.npz')
                try:
                    assert os.path.exists(path)
                except:
                    break
                data_dict = np.load(path, allow_pickle=True)
                radar_data = data_dict['data']
                miss_dates = data_dict['miss_dates']
                # print(miss_dates)
                ## index produced by miss_index
                miss_index = [date_to_index(miss_date) for miss_date in miss_dates]
                ## complement data
                comp_radar_data = complete_miss_dates_in_data(radar_data, miss_index)
                ## exclude invalid index
                invalid_index = []
                for i in range(comp_radar_data.shape[0]):
                    if comp_radar_data[i][:sub_region, :sub_region].max() == 255:
                        invalid_index.append(i)
                comp_radar_data = comp_radar_data[:, :sub_region, :sub_region]
                ## gen data
                total_miss_index = miss_index + invalid_index
                total_miss_index.sort() 
                valid_continue_radar_frames = split_data_to_continue_frames(comp_radar_data, total_miss_index, comp_radar_data.shape[0]-len(total_miss_index))
                # print(f'{y}-{m:02d}-{p} has {len(valid_continue_radar_frames)} valid continue frames and has total_miss_index: {total_miss_index}')

                for continue_frames in valid_continue_radar_frames:
                    event_set = event_filtering(continue_frames, pixel_threshold=pixel_threshold, l_in=l_in, l_out=l_out)
                    if len(event_set) == 0:
                        continue
                    
                    ## upload event ##
                    for event in event_set:
                        upload_event(event, event_id)
                        print(f'{event_id}.npy')
                        event_id += 1
    

    split = 'valid'
    print('valid split:')
    for y in [2018]:
        for m in range(1, 7):
            for p in range(1, 4):
                path = os.path.join('/mnt/lustre/gongjunchao/meteonet_eg',
                        f'SE_reflectivity_old_product_{y}', 
                        f'reflectivity-old-SE-{y}-{m:02d}',
                        f'reflectivity_old_SE_{y}_{m:02d}.{p}.npz')
                try:
                    assert os.path.exists(path)
                except:
                    break
                data_dict = np.load(path, allow_pickle=True)
                radar_data = data_dict['data']
                miss_dates = data_dict['miss_dates']
                # print(miss_dates)
                ## 由miss_index产生的index
                miss_index = [date_to_index(miss_date) for miss_date in miss_dates]
                ## 补齐数据
                comp_radar_data = complete_miss_dates_in_data(radar_data, miss_index)
                ## 筛选出(400, 400)区域内，因雷达未扫描到产生的无效index
                invalid_index = []
                for i in range(comp_radar_data.shape[0]):
                    if comp_radar_data[i][:sub_region, :sub_region].max() == 255:
                        invalid_index.append(i)
                comp_radar_data = comp_radar_data[:, :sub_region, :sub_region]
                ## 生成有效的数据
                total_miss_index = miss_index + invalid_index
                total_miss_index.sort() ##由小到大重排序
                valid_continue_radar_frames = split_data_to_continue_frames(comp_radar_data, total_miss_index, comp_radar_data.shape[0]-len(total_miss_index))
                # print(f'{y}-{m:02d}-{p} has {len(valid_continue_radar_frames)} valid continue frames and has total_miss_index: {total_miss_index}')

                for continue_frames in valid_continue_radar_frames:
                    event_set = event_filtering(continue_frames, pixel_threshold=pixel_threshold, l_in=l_in, l_out=l_out)
                    if len(event_set) == 0:
                        continue
                    
                    ## upload event ##
                    for event in event_set:
                        upload_event(event, event_id)
                        print(f'{event_id}.npy')
                        event_id += 1
    
    split = 'test'
    print('test split:')
    for y in [2018]:
        for m in range(7, 13):
            for p in range(1, 4):
                path = os.path.join('/mnt/lustre/gongjunchao/meteonet_eg',
                        f'SE_reflectivity_old_product_{y}', 
                        f'reflectivity-old-SE-{y}-{m:02d}',
                        f'reflectivity_old_SE_{y}_{m:02d}.{p}.npz')
                try:
                    assert os.path.exists(path)
                except:
                    break
                data_dict = np.load(path, allow_pickle=True)
                radar_data = data_dict['data']
                miss_dates = data_dict['miss_dates']
                # print(miss_dates)
                ## 由miss_index产生的index
                miss_index = [date_to_index(miss_date) for miss_date in miss_dates]
                ## 补齐数据
                comp_radar_data = complete_miss_dates_in_data(radar_data, miss_index)
                ## 筛选出(400, 400)区域内，因雷达未扫描到产生的无效index
                invalid_index = []
                for i in range(comp_radar_data.shape[0]):
                    if comp_radar_data[i][:sub_region, :sub_region].max() == 255:
                        invalid_index.append(i)
                comp_radar_data = comp_radar_data[:, :sub_region, :sub_region]
                ## 生成有效的数据
                total_miss_index = miss_index + invalid_index
                total_miss_index.sort() ##由小到大重排序
                valid_continue_radar_frames = split_data_to_continue_frames(comp_radar_data, total_miss_index, comp_radar_data.shape[0]-len(total_miss_index))
                # print(f'{y}-{m:02d}-{p} has {len(valid_continue_radar_frames)} valid continue frames and has total_miss_index: {total_miss_index}')

                for continue_frames in valid_continue_radar_frames:
                    event_set = event_filtering(continue_frames, pixel_threshold=pixel_threshold, l_in=l_in, l_out=l_out)
                    if len(event_set) == 0:
                        continue
                    
                    ## upload event ##
                    for event in event_set:
                        upload_event(event, event_id)
                        print(f'{event_id}.npy')
                        event_id += 1
    

### srun -p ai4earth --kill-on-bad-exit=1 --quotatype=reserved --gres=gpu:0 python -u preprocess_upload.py ###