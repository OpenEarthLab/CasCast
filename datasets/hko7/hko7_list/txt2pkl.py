import pickle

txt_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/datasets/hko7_list/hko7_full_list.txt'

with open(txt_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

lines_list = [line.strip() for line in lines]


with open('/mnt/cache/gongjunchao/workdir/radar_forecasting/datasets/hko7_list/hko7_full_list.pkl', 'wb') as pkl_file:
    pickle.dump(lines_list, pkl_file)
