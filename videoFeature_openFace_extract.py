import os
import numpy as np
import pickle
import json
import csv

def getNo(strname):
    index = 0
    for i in range(len(strname)):
        if i == 0:
            continue
        else:
            if not (strname[i] >= '0' and strname[i] <= '9'):
                index = i
                break
    return strname[:index]

# 获取去除后缀后的文件名
def getName(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

def load_readtime_data(readtime_path):
    readtime_data = {}
    with open(readtime_path, "r", encoding="utf8") as file:
        data = json.load(file)
        readtime_data.update(data)
    return readtime_data

def write_features_to_csv(features, file_path, mode='a'):
    with open(file_path, mode, newline='') as file:
        writer = csv.writer(file)
        writer.writerows(features)

def csv_to_pkl(csv_file_path, pkl_file_path):
    video_feature = {}
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            video_name = row[0]
            features = [float(value) for value in row[1:]]
            if video_name in video_feature:
                video_feature[video_name].append(features)
            else:
                video_feature[video_name] = [features]
    with open(pkl_file_path, 'wb') as f1:
        pickle.dump(video_feature, f1, protocol=pickle.HIGHEST_PROTOCOL)

csv_root_path0 = '/mnt/Openface_abnorm'
csv_root_path1 = '/mnt/Openface_norm'
csv_root_path2 = '/mnt/Openface_adalt'

readtime_path0 = '/home/lixiyang/documents/Medical_P/readtime_abnorm.json'
readtime_path1 = '/home/lixiyang/documents/Medical_P/readtime_norm.json'
readtime_path2 = '/home/lixiyang/documents/Medical_P/readtime_adalt.json'

video_feature_csv0 = '/home/lixiyang/documents/Medical_P/feature/abnorm/video_feature.csv'
video_feature_csv1 = '/home/lixiyang/documents/Medical_P/feature/norm/video_feature.csv'
video_feature_csv2 = '/home/lixiyang/documents/Medical_P/feature/adalt/video_feature.csv'

video_feature_pkl0 = '/home/lixiyang/documents/Medical_P/feature/abnorm/video_feature.pkl'
video_feature_pkl1 = '/home/lixiyang/documents/Medical_P/feature/norm/video_feature.pkl'
video_feature_pkl2 = '/home/lixiyang/documents/Medical_P/feature/adalt/video_feature.pkl'

timetamps_path0 = '/home/lixiyang/documents/Medical_P/timetamps/abnorm/video_timetamps.pkl'
timetamps_path1 = '/home/lixiyang/documents/Medical_P/timetamps/norm/video_timetamps.pkl'
timetamps_path2 = '/home/lixiyang/documents/Medical_P/timetamps/adalt/video_timetamps.pkl'

csv_name = os.listdir(csv_root_path1)

video_feature = {}
json_readtime = load_readtime_data(readtime_path1)

QAtime = pickle.load(
    open(timetamps_path1, 'rb')
)

# 从JSON数据中提取文件名，并存储在fnos列表中
fnos = [getName(i) for i in json_readtime.keys()]

# 创建包含json_readtime和QAtime所有keys的交集
combined_keys = set(fnos).intersection(set(getName(i) for i in QAtime.keys()))

cnt = 0
for i in csv_name:
    if getName(i) not in combined_keys:
        continue
    cnt += 1
    frames_feature_list = [[] for _ in range(22)]
    csv_path = os.path.join(csv_root_path1, i)
    with open(csv_path, encoding='latin1') as csvfile:
        readcsv = csv.reader(csvfile)
        readcsv = list(readcsv)
        fname = getName(i)
        timetamps = [[5, 90], [110, 190], [205, 320]]
        timetamps.append(json_readtime[fname][0:2])
        timetamps.append(json_readtime[fname][2:4])
        timetamps.append(json_readtime[fname][4:6])
        for j in QAtime[fname]:
            timetamps.append(j)
        partNo = 0
        starttime = timetamps[partNo][0]
        endtime = timetamps[partNo][1]
        count = timetamps[partNo][0]
        for j in readcsv[1:]:
            if len(j) < 5:  # 检查行长度
                print(f"Skipping line in {i} due to insufficient columns: {j}")
                continue
            if float(j[2]) > endtime:
                partNo += 1
                if partNo >= len(timetamps):
                    print(f"Warning: partNo {partNo} exceeded timetamps length {len(timetamps)} in file {i}")
                    break
                starttime, endtime = timetamps[partNo]
                count = starttime
            if partNo < len(frames_feature_list) and float(j[2]) >= count and int(j[4]) == 1:
                j = list(map(float, j))
                if partNo >= len(timetamps):
                    print(f"Warning: partNo {partNo} exceeded timetamps length {len(timetamps)} in file {i}")
                    break
                frames_feature_list[partNo].append(j[-35:])
                print(j[-35:])
                count = j[2] + (1 if partNo <= 5 else 0.5)
    video_feature[os.path.splitext(i)[0]] = frames_feature_list
    if cnt % 10 == 0:
        with open(f'/home/lixiyang/documents/Medical_P/feature/norm/video_feature{cnt}.pkl', 'wb') as f:
            pickle.dump(video_feature, f, protocol=pickle.HIGHEST_PROTOCOL)
        video_feature = {}

if video_feature:
    with open(f'/home/lixiyang/documents/Medical_P/feature/norm/video_feature{cnt}.pkl', 'wb') as f:
        pickle.dump(video_feature, f, protocol=pickle.HIGHEST_PROTOCOL)
