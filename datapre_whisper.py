from difflib import SequenceMatcher
import time
from moviepy.editor import *
import os
import torch
import json
from pydub import AudioSegment
from zhconv import convert
# 音视频转文字模型的加载
# 音视频转文字模型的加载
import whisper
import csv
import logging
from multiprocessing import Pool, cpu_count

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/lixiyang/documents/Medical_P/log/process_log.log"),
        logging.StreamHandler()
    ]
)

# 视频时长裁剪
def video_clip(video_root_path, video_save_path, audio_root_path, video_name):
    logging.info("开始视频裁剪: %s", video_name)
    try:
        if os.path.splitext(video_name)[-1] == '.mp4':
            video_path = os.path.join(video_root_path, video_name)
            video = VideoFileClip(video_path)
            video_clip = video
            if video.duration > 316:
                video_clip = video.subclip(316)

            video_save_name = video_name.split('.')[0] + "part.mp3"
            video_save_path = os.path.join(video_save_path, video_save_name)
            video_clip.audio.write_audiofile(f'{video_save_path}')
            video_clip.close()
            logging.info("视频裁剪完成: %s", video_name)
        elif os.path.splitext(video_name)[-1] == '.mp3':
            audio_path = os.path.join(audio_root_path, video_name)
            song = AudioSegment.from_mp3(audio_path)
            video_save_name = video_name.split('.')[0] + "part.mp3"
            video_save_path = os.path.join(video_save_path, video_save_name)
            song[316*1000:].export(video_save_path)
            logging.info("音频裁剪完成: %s", video_name)
    except Exception as e:
        logging.error("视频裁剪错误: %s, 错误信息: %s", video_name, e)

# 音视频转文字
def ToText(video_save_path, video_name, audio_txt_path):
    logging.info("开始转文字: %s", video_name)
    video_path = os.path.join(video_save_path, video_name.split('.')[0]+"part.mp3")
    
    try:
        start =time.time()
        result = model.transcribe(video_path)
        for i in result['segments']:
            i['text'] = convert(i['text'], 'zh-hans')
        end = time.time()
        logging.info('%s to text Running Time: %f Seconds', video_name, end - start)

        output_file = os.path.join(audio_txt_path, video_name.split('.')[0] + '.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            for segment in result['segments']:
                f.write(segment['text'] + '\n')

        logging.info("转文字完成: %s", video_name)
    except Exception as e:
        logging.error("转文字错误: %s, 错误信息: %s", video_name, e)
    return result

# 计算文本相似度
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# 提取关键问题的时间戳
# 关键问题如下定义
def extract_key_question_time(result):
    questions = [
        "请你尽可能全面准确的回答 今天过得怎么样",
        "家乡是哪里的",
        "最喜欢你家乡的哪些美食 景点",
        "家人同事同学朋友 关系处得怎么样",
        "性格内向还是外向一些",
        "最近两周心情怎么样",
        "目前的学习或工作的兴趣如何",
        "容不容易责备自己 感到自己连累了其他人",
        "觉得自己的行动思考或说话都比较迟钝",
        "是否经常感到紧张焦虑担心惶恐不安",
        "有没有哪段时间 感到兴奋或亢奋或者精力旺盛",
        "有没有哪段时间 连续几天持续地感到烦躁易怒,以至于争论吵架或打架或者对着外人大吼",
        "有没有哪段时间 你总喜欢滔滔不绝地讲话，说话快得让人难以理解",
        "有没有哪段时间 你觉得自己思维比以往格外活跃脑子格外聪明",
        "有没有哪段时间 你认为有人在暗中监视你 故意议论你或企图伤害你吗",
        "有没有哪段时间 你能听到其他人不能听到的声音或者看到别人看不到的东西 有的话 请你仔细讲一下",
        "谢谢你的参与 再见"
    ]
    # 关键问题时间戳
    key_question_time = []
    
    j = 0
    for i in result['segments']:
        if similarity(i['text'], questions[j]) > 0.4 or similarity(i['text'], '有没有哪段时间') > 0.5:
            if similarity(i['text'], '有没有哪段时间') > 0.5 and similarity(result['segments'][result['segments'].index(i)+1]['text'], questions[j]) > 0.4:
                # logging.info("!!!!!",result['segments'][result['segments'].index(i)+1]['text'])
                continue            
            time_dict = {'id':-1,'start':0.0, 'end':0.0}
            time_dict['id'] = i['id']
            time_dict['start'] = i['start'] + 316
            time_dict['end'] = i['end'] + 316
            key_question_time.append(time_dict)
            j = j + 1
        if j > 16:
            break

    # logging.info(key_question_time)
    # logging.info(len(key_question_time))
    return key_question_time


# 关键问题的回答
# logging.info(key_question_time)
# 16个问题的回答, 字符串列表
def integrate_key_question_answer(result, key_question_time):
    key_question_answer = []
    i = 0
    while i < len(key_question_time) - 1:
        text = ''
        for j in range(key_question_time[i]['id'] + 1, key_question_time[i+1]['id']):
            if similarity(result['segments'][j]['text'], "有的话请你仔细讲一下") > 0.4 or similarity(result['segments'][j]['text'], "可以仔细讲一下") > 0.4:
                text = ''
            elif similarity(result['segments'][j]['text'], "有没有哪段时间") > 0.5 or similarity(result['segments'][j]['text'], "好的") > 0.8 or similarity(result['segments'][j]['text'], "今天过得怎么样") > 0.7:
                continue
            elif text == '':
                text = result['segments'][j]['text']
            else:
                text = text +',' + result['segments'][j]['text']
        key_question_answer.append(text)
        i = i + 1
    # logging.info(key_question_answer)
    # logging.info(len(key_question_answer))
    # 将回答整合到时间戳字典列表中
    key_question_time.append({'answer':key_question_answer})
    return key_question_time
# 视频转音频
def video2audio(video_root_path, audio_root_path, videoname):
    try:
        video_path = os.path.join(video_root_path, videoname)
        my_audio_clip = AudioFileClip(video_path)
        audio_path = os.path.join(audio_root_path, videoname.split('.')[0] + ".mp3")
        my_audio_clip.write_audiofile(audio_path)
    except Exception as e:
        logging.error(f'Error converting video to audio for {videoname}: {e}')
        return False
    return True


def extract_time(result):
    text1 = [
        "盼啊 盼啊",
        "眼看春节就快到了",
        "想到这",
        "想到这 我不由得笑了起来",
        "我不由得笑了起来",
        "在春节前",
        "在春节前,人们个个喜气洋洋,个个精神饱满",
        "人们个个喜气洋洋",
        "个个精神饱满",
        "逛街的人络绎不绝",
        "有的在买年画",
        "有的在买年货",
        "逛街的人络绎不绝 有的在买年画 有的在买年货",
        "有的围着火炉看电视",
        "还有的人在打麻将",
        "打扑克等等",
        "不一而足",
        "大年三十",
        "人们常常玩到深夜",
        "大年三十 人们常常玩到深夜",
        "嘴里啃着美味水果",
        "手里燃放烟花爆竹",
        "大人小孩都载歌载舞",
        "忘情地玩个痛快",
    ]

    text2 = [
        "卢沟桥位于北京广安门外的永定河上",
        "距离天安门15千米",
        "它始建于金代大定年间",
        "历时三年建成",
        "定名为广利桥",
        "又因永定河旧称卢沟河",
        "所以广利桥俗称卢沟桥",
        "卢沟桥是北京地区现存的最古老的一座连拱石桥",
        "明清两代都有重修",
        "现在所见到的为1986年重修复原后的石桥",
        "桥长266.5米",
        "桥面宽9.3米",
        "为花岗岩所砌成",
    ]

    text3 = [
        "你们把眼睛凑近去细察人生吧",
        "从各个方面看",
        "我们都会感到人的一生处处是惩罚",
        "每天都有大的烦恼或小的操心",
        "昨天你曾为一个亲人的健康担心",
        "今天又为自己的健康担忧",
        "明天将是金钱方面的麻烦",
        "后天又将受到一桩诽谤",
        "还不去算内心的种种痛苦",
        "没完没了",
        "散了一片乌云",
        "又来一片乌云",
        "一百天里",
        "难得有一天是充满欢乐和阳光的",
    ]

    betime1 = 0
    endtime1 = 0
    betime2 = 0
    endtime2 = 0
    betime3 = 0
    endtime3 = 0
    for i in result['segments']:
        ttext = i['text']
        if i['start'] > 224:
            break
        if betime1 == 0:
            for j in text1:
                if similarity(ttext, j) > 0.4:
                    betime1 = i['start'] + 316
        if betime1 != 0 and betime2 == 0:
            for j in text1:
                if similarity(ttext, j) > 0.4:
                    endtime1 = i['end'] + 316
        if betime2 == 0 and endtime1 != 0:
            for j in text2:
                if similarity(ttext, j) > 0.4:
                    betime2 = i['start'] + 316
        if betime2 != 0 and betime3 == 0:
            for j in text2:
                if similarity(ttext, j) > 0.4:
                    endtime2 = i['end'] + 316
        if betime3 == 0 and endtime2 != 0:
            for j in text3:
                if similarity(ttext, j) > 0.4:
                    betime3 = i['start'] + 316
        if betime3 != 0:
            for j in text3:
                if similarity(ttext, j) > 0.4:
                    if int(i['end']) + 316 - betime3 > 70:
                        break
                    endtime3 = i['end'] + 316
    betime1 = int(betime1)
    betime2 = int(betime2)
    betime3 = int(betime3)
    endtime1 = int(endtime1)
    endtime2 = int(endtime2)
    endtime3 = int(endtime3)
    timelist = []
    timelist.append(betime1)
    timelist.append(endtime1)
    timelist.append(betime2)
    timelist.append(endtime2)
    timelist.append(betime3)
    timelist.append(endtime3)
    return timelist
# 得到无后缀文件名
def get_filenames_without_extension(directory):
    filenames = os.listdir(directory)
    result = []
    for filename in filenames:
        name, _ = os.path.splitext(filename)
        result.append(name)
    return result
# 得到文件编号
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


def process_videos(video_root_path, audio_root_path, video_save_path, txt_audio_path, video_json_path, error_save_path, readtime_path):
    # Check all directories before processing
    if not (check_directory(video_root_path) and 
            check_directory(audio_root_path) and 
            check_directory(video_save_path) and 
            check_directory(txt_audio_path) and 
            check_directory(video_json_path) and 
            check_directory(error_save_path)):
        logging.error('One or more directories do not exist. Exiting.')
        exit(1)
    # 错误编号列表
    # error_no = ['P644', 'N1213', 'P647', 'P674', 'P090', 'P377', 'P469', 'P479', 'P481', 'P483', 'P484', 'P485', 'P486', 'P488', 'P489', 'P491', 'P496', 'P497', 'P498', 'P500', 'P503', 'P504', 'P505', 'P506', 'P694', 'P596', 'P602', 'P655', 'P766', 'P655', 'P826', 'P482', 'P493', 'P673']

    # 读取目录下的所有文件名
    fs = os.listdir(video_root_path)

    timetamp = {}  # 读文字时间戳

    for f in fs:
        if f.split('.')[0]:
            logging.info("Processing file: %s", f)
            try:
                video2audio(video_root_path, audio_root_path, f)
                video_clip(video_root_path, video_save_path, audio_root_path, f)
                result = ToText(video_save_path, f, txt_audio_path)
                key_question_time = extract_key_question_time(result)

                if len(key_question_time) != 17:
                    logging.warning("问题数不满17: %s", f)
                    with open(error_save_path, "a+", encoding="utf8", newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([f, "问题数不满17"])
                else:
                    json_str = json.dumps(key_question_time, ensure_ascii=False, indent=1)
                    json_save_path = os.path.join(video_json_path, f.split('.')[0] + ".json")
                    with open(json_save_path, "w", encoding="utf8", newline='\n') as file:
                        file.write(json_str)
                    logging.info("Q&A处理成功: %s", f)

                tl = extract_time(result)
                timetamp[f.split('.')[0]] = tl
                if 0 in tl:
                    logging.warning("时间戳包含0: %s", f)
                    with open(error_save_path, "a+", encoding="utf8", newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([f, "时间戳包含0"])
                else:
                    if (len(timetamp) % 10 == 0):
                        json_str1 = json.dumps(timetamp, ensure_ascii=False, indent=1)
                        with open(readtime_path, "a+", encoding="utf8", newline='\n') as file:
                            file.write(json_str1)
                        timetamp = {}
                    logging.info("read处理成功: %s", f)
            except Exception as e:
                logging.error("处理文件出错: %s, 错误信息: %s", f, e)

    if timetamp:
        json_str1 = json.dumps(timetamp, ensure_ascii=False, indent=1)
        with open(readtime_path, "a+", encoding="utf8", newline='\n') as file:
            file.write(json_str1)

def check_directory(path):
    if not os.path.exists(path):
        logging.error(f'Directory does not exist: {path}')
        return False
    return True

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_VISIBLE_DEVICES'] = "1,0,2,3,4,5,6,7"
torch.cuda.set_device(5)
model = whisper.load_model("medium")
device=torch.device ( "cuda" if torch.cuda.is_available () else "cpu")
model.to(device)


device=torch.device ( "cuda:5" if torch.cuda.is_available () else "cpu")
# 原始视频存放路径
video_root_path0 = "/mnt/abnorm"
video_root_path1 = "/mnt/norm"
video_root_path2 = "/mnt/adalt"

# 原始视频转换为音频存放路径
audio_root_path0 = "/home/lixiyang/documents/Medical_P/audio/abnorm"
audio_root_path1 = "/home/lixiyang/documents/Medical_P/audio/norm"
audio_root_path2 = "/home/lixiyang/documents/Medical_P/audio/adalt"

# 视频剪辑后存放路径
video_save_path0 = "/home/lixiyang/documents/Medical_P/videopart/abnorm"
video_save_path1 = "/home/lixiyang/documents/Medical_P/videopart/norm"
video_save_path2 = "/home/lixiyang/documents/Medical_P/videopart/adalt"

# 文字存放路径
txt_audio_path0 = '/home/lixiyang/documents/Medical_P/audiotxt/abnorm'
txt_audio_path1 = '/home/lixiyang/documents/Medical_P/audiotxt/norm'
txt_audio_path2 = '/home/lixiyang/documents/Medical_P/audiotxt/adalt'

# 提取
video_json_path0 = "/home/lixiyang/documents/Medical_P/videojson/abnorm"
video_json_path1 = "/home/lixiyang/documents/Medical_P/videojson/norm"
video_json_path2 = "/home/lixiyang/documents/Medical_P/videojson/adalt"

# 视频处理错误存放路径
error_save_path0 = "/home/lixiyang/documents/Medical_P/error/abnorm/error_abnorm.csv"
error_save_path1 = "/home/lixiyang/documents/Medical_P/error/norm/error_norm.csv"
error_save_path2 = "/home/lixiyang/documents/Medical_P/error/adalt/error_adalt.csv"

readtime_path0 = '/home/lixiyang/documents/Medical_P/readtime_abnorm.json'
readtime_path1 = '/home/lixiyang/documents/Medical_P/readtime_norm.json'
readtime_path2 = '/home/lixiyang/documents/Medical_P/readtime_adalt.json'
