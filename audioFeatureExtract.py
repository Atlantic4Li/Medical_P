import csv
import opensmile
import pickle
from pydub import AudioSegment
from moviepy.editor import *
import os
import json
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_VISIBLE_DEVICES'] = "5,2,3,4,1,6,7,0"

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,# FeatureSet.emobase # FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)


# 提取音频特征
def audio_extract(audio_name, que_ans_time, audio_root_path, readtime):
    videotime = [[5, 90], [110, 190], [205, 320]]
    timetamps = videotime + [readtime[0:2], readtime[2:4], readtime[4:6]] + que_ans_time
    audio_path = os.path.join(audio_root_path, audio_name + ".mp3")
    sound = AudioSegment.from_mp3(audio_path)
    ado_feature = []
    for i in timetamps:
        start_time = int(i[0] * 1000)
        end_time = int(i[1] * 1000)
        if end_time <= start_time:
            end_time = start_time + 100
        word = sound[start_time:end_time]
        tmp_path = os.path.join('audio_tmp_' + audio_name + ".mp3")
        word.export(tmp_path, format="mp3")
        if not os.path.exists(tmp_path):
            continue
        try:
            y = smile.process_file(tmp_path)
            ado_feature.append(y.values[0])
        except Exception as e:
            print(f"处理文件 '{tmp_path}' 时出错: {e}")
            os.remove(tmp_path)
            continue
        os.remove(tmp_path)
    return ado_feature

# 加载 readtime 数据
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

audio_root_path0 = "/home/lixiyang/documents/Medical_P/audio/abnorm"
audio_root_path1 = "/home/lixiyang/documents/Medical_P/audio/norm"
audio_root_path2 = "/home/lixiyang/documents/Medical_P/audio/adalt"

readtime_path0 = '/home/lixiyang/documents/Medical_P/readtime_abnorm.json'
readtime_path1 = '/home/lixiyang/documents/Medical_P/readtime_norm.json'
readtime_path2 = '/home/lixiyang/documents/Medical_P/readtime_adalt.json'

timetamps_path0 = '/home/lixiyang/documents/Medical_P/timetamps/abnorm/audio_timestamps.pkl'
timetamps_path1 = '/home/lixiyang/documents/Medical_P/timetamps/norm/audio_timestamps.pkl'
timetamps_path2 = '/home/lixiyang/documents/Medical_P/timetamps/adalt/audio_timestamps.pkl'

audio_feature_path0 = '/home/lixiyang/documents/Medical_P/feature/abnorm/audio_feature.csv'
audio_feature_path1 = '/home/lixiyang/documents/Medical_P/feature/norm/audio_feature.csv'
audio_feature_path2 = '/home/lixiyang/documents/Medical_P/feature/adalt/audio_feature.csv'

timetamps = pickle.load(open(timetamps_path1, "rb"), encoding="latin1")
json_readtime = load_readtime_data(readtime_path1)

audio_feature = {}
cnt = 0

for i in timetamps.keys():
    cnt += 1
    if i not in json_readtime:
        print(f"Missing readtime data for {i}")
        continue
    
    audio_feature[i] = audio_extract(i, timetamps[i], audio_root_path1, json_readtime[i])
    
    if cnt % 50 == 0 and cnt != 0:
        with open(f'/home/lixiyang/documents/Medical_P/feature/norm/audio_feature{cnt}.pkl', 'wb') as f1:
            pickle.dump(audio_feature, f1, protocol=pickle.HIGHEST_PROTOCOL)
        audio_feature = {}

if audio_feature:
    with open(f'/home/lixiyang/documents/Medical_P/feature/norm/audio_feature{cnt}.pkl', 'wb') as f1:
        pickle.dump(audio_feature, f1, 0)
