from keras.models import load_model
import numpy as np
import re
import jieba
#from keras.preprocessing.sequence import pad_sequences

read_dictionary = np.load('word.npy', allow_pickle=True).item()
dictionary = {key: value for key, value in read_dictionary.items() if value<=100000}
model = load_model('weights.010-0.9609.hdf5')

MAX_SEQUENCE_LENGTH = 50
def predict(text):
    txt = remove_punctuation(text)
    txt = [w for w in list(jieba.cut(txt))][:MAX_SEQUENCE_LENGTH]
    #tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    seq = []
    for word in txt:
        if word in dictionary:
            seq.append(dictionary[word])
        else:
            seq.append(0)
    while len(seq) < MAX_SEQUENCE_LENGTH:
        seq.append(0)
    padded = seq
    pred = model.predict(padded)
    cat_id= pred.argmax(axis=1)[0]
    if cat_id == 0:
        cat = '秦'
    elif cat_id ==1:
        cat = '汉、三国'
    elif cat_id ==2:
        cat = '魏晋南北朝'
    elif cat_id ==3:
        cat = '隋唐五代'
    elif cat_id ==4:
        cat = '宋、金'
    elif cat_id ==5:
        cat = '元'
    elif cat_id ==6:
        cat = '明'
    else:
        cat = '清'
    return cat

#定义删除除字母,数字，汉字以外的所有符号的函数

def remove_punctuation(line):
    line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('',line)
    return line

print(predict('青青园中葵，朝露待日晞'))
print(predict('我愿平东海，身沉心不改。'))
print(predict('大漠孤烟直'))
print(predict('白毛浮绿水'))
print(predict('采菊东篱下，悠然现南山'))
print(predict('大风起兮云飞扬'))
print(predict('明月松间照，清泉石上流'))
print(predict('琼姿只合在瑶台，谁向江南处处栽'))
print(predict('一蓑一笠一扁舟，一丈丝纶一寸钩'))
print(predict('东临碣石，以观沧海'))
