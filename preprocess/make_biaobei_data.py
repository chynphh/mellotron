# example for filelists
# data/example1.wav|exploring the expanses of space to keep our planet safe|1
# data/example2.wav|and all the species that call it home|1
# seperate train and val in different filelists

# example for speaker info
# ; Some pipe(|) separated metadata about all LibriVox readers, whose work was used
# ; in the corpus.
# ;
# ; The meaning of the fields in left-to-right order is as follows:
# ;
# ; reader_id: the ID of the reader in the LibriVox's database
# ; gender: 'F' for female, 'M' for male
# ; subset: the corpus subset to which the reader's audio is assigned
# ; duration: total number of minutes of speech by the reader, included in the corpus
# ; name: the name under which the reader is registered in LibriVox
# ;
# ;ID  |SEX| SUBSET           |MINUTES| NAME
# 14   | F | train-clean-360  | 25.03 | Kristin LeMoine
# 16   | F | train-clean-360  | 25.11 | Alys AtteWater
# 17   | M | train-clean-360  | 25.04 | Gord Mackenzie
# 19   | F | train-clean-100  | 25.19 | Kara Shallenberg
import os
from tqdm import tqdm
import random
random.seed(951357)
text_data = '/data/bznsyp/ProsodyLabeling/000001-010000.txt'

data_root = "/data/bznsyp/Wave/"
with open(text_data) as f:
    lines = f.readlines()

num = len(lines) // 2
datas = []
for i in range(num):
    wav_id = lines[i*2].strip().split('\t')[0]
    # print(lines[i*2].strip().split("\t"))
    # print(lines[i*2+1].strip().split("\t"))
    data = f"{data_root}{wav_id}.wav|{lines[i*2+1].strip()}|600\n"
    datas.append(data)
    # print(data)

# # get all speaker info and data
# speaker_infos = {}
# datas = {}
# for speaker_folder in tqdm(speaker_folders):
#     speaker_name = speaker_folder[-5:]
#     for file in os.listdir(speaker_folder):
#         if file.endswith(".pinyin"):
#             basename = file[:-7]
#             if speaker_name not in speaker_infos:
#                 speaker_info = {}
#                 speaker_info['id'] = len(speaker_infos)
#                 with open(os.path.join(speaker_folder, f"{basename}.metadata")) as f:
#                     lines = f.readlines()
#                     speaker_info['name'] = lines[20].strip().split("\t")[1]
#                     speaker_info['sex'] = lines[21].strip().split("\t")[1]
#                     speaker_info['age'] = lines[22].strip().split("\t")[1]
#                     speaker_info['bir'] = lines[25].strip().split("\t")[1]
#                     # print(speaker_info)
#                 speaker_infos[speaker_name] = speaker_info
#                 datas[speaker_name] = []
#             wav_name = os.path.join(speaker_folder, basename+".wav") 
#             with open(os.path.join(speaker_folder, file)) as f:
#                 text = f.readline().strip()
#             data = f"{wav_name}|{text}|{speaker_infos[speaker_name]['id']}\n"
#             datas[speaker_name].append(data)

# print(len(speaker_name))

# spilt data and wirte
train_data = []
val_data = []
random.shuffle(datas)
train_data += datas[:int(len(datas)*0.9)]
val_data += datas[int(len(datas)*0.9):]

with open("../filelists/biaobei_train_filelist.txt", "w+") as f:
    f.writelines(train_data)
with open("../filelists/biaobei_val_filelist.txt", "w+") as f:
    f.writelines(val_data)

# # write speaker info
# # 19   | F | train-clean-100  | 25.19 | Kara Shallenberg
# infos = []
# for speaker_info in speaker_infos.values():
#     info = f"{speaker_info['id']}|{speaker_info['sex'][0]}|-1|{speaker_info['name']}\n"
#     infos.append(info)
# with open("../filelists/aidatatang_speakerinfo.txt", "w+") as f:
#     f.writelines(infos)
