from difflib import SequenceMatcher
import os
import time
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_VISIBLE_DEVICES'] = "5,2,3,4,1,6,7,0"
import torch
torch.cuda.set_device(6)
from sentence_transformers import SentenceTransformer
import json
import pickle
from zhconv import convert
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/lixiyang/documents/Medical_P/log/process_norm.log"),
        logging.StreamHandler()
    ]
)

device = torch.device("cuda:6" if torch.cuda.is_available () else "cpu")

model = SentenceTransformer("/home/lixiyang/model/KoichiYasuoka/chinese-bert-wwm-ext-upos")
model.to(device)

text_root_path0 = "/home/lixiyang/documents/Medical_P/audiotxt/abnorm"
text_root_path1 = "/home/lixiyang/documents/Medical_P/audiotxt/norm"
text_root_path2 = "/home/lixiyang/documents/Medical_P/audiotxt/adalt"

sentence_save_path0 = '/home/lixiyang/documents/Medical_P/feature/abnorm/sentenceFeature.pkl'
sentence_save_path1 = '/home/lixiyang/documents/Medical_P/feature/norm/sentenceFeature.pkl'
sentence_save_path2 = '/home/lixiyang/documents/Medical_P/feature/adalt/sentenceFeature.pkl'

def ToText(video_save_path, video_name, audio_txt_path):
    logging.info("开始转文字: %s", video_name)
    try:
        start =time.time()
        video_path = os.path.join(video_save_path, video_name.split('.')[0]+"part.mp3")
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

def text_extract(text_root_path, text_name):
    json_path = os.path.join(text_root_path, text_name + ".json")
    #print(json_path)
    sentence_embeddings = []
    sentences = []
    with open(json_path, "r", encoding="utf8") as file:
        json_text = json.load(file)
        timetamp = []
        for i in range(16):
            timetamp.append([json_text[i]['start'], json_text[i+1]['start']])
        for i in json_text[17]['answer']:
            sentences.append(i)
            sentence_embedding = model.encode(i)
            sentence_embeddings.append(sentence_embedding)
        #print(len(sentence_embedding))
        return timetamp, sentences, sentence_embeddings # 16*1, 16*1, 16*1*768
    
def get_filenames_without_extension(text_root_path):
    filenames = os.listdir(text_root_path)
    result = []
    for filename in filenames:
        name, _ = os.path.splitext(filename)
        json_path = os.path.join(text_root_path, name + ".json")
        with open(json_path, "r", encoding="utf8") as file:
            json_text = json.load(file)
            if len(json_text) == 18:
                result.append(name)
    return result

def the_main(text_root_path, sentence_save_path):
    fns = get_filenames_without_extension(text_root_path)
    sentence = {} 
    sentence_embedding = {} 
    timetamps = {} 
    for f in fns:
        timetamp, stc, stc_embedding = text_extract(text_root_path, f)
        sentence[f] = stc
        sentence_embedding[f] = stc_embedding
        timetamps[f] = timetamp
    with open(sentence_save_path, 'wb') as f1:
        pickle.dump(sentence, f1, 0)
    print("文本特征提取完毕")

the_main(text_root_path0, sentence_save_path0)
# the_main(text_root_path1, sentence_save_path1)
# the_main(text_root_path2, sentence_save_path2)


