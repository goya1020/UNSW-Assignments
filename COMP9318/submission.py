import pandas as pd
import numpy as np
import nltk
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import pickle
#from sklearn.metrics import f1_score


# This function makes data dictionary.
def making_data_dic_N_part_of_speech(raw_data):
    part_of_speech_index = {'CC':0,'CD':1,'DT':2,'EX':3,'FW':4,'IN':5,'JJ':6,'JJR':7,'JJS':8,'LS':9,'MD':10,'NN':11,'NNS':12,'NNP':13,'NNPS':14,'PDT':15,'POS':16,'PRP':17,'PRP$':18,'RB':19,'RBR':20,'RBS':21,'RP':22,'SYM':23,'TO':24,'UH':25,'VB':26,'VBD':27,'VBG':28,'VBN':29,'VBP':30,'VBZ':31,'WDT':32,'WP':33,'WP$':34 ,'WRB':35}
    data_dict = OrderedDict()
    data_dict_with_stress_number = OrderedDict()
    tag_indicator_list = []
    for d in raw_data:
        dd = d.split(':')
        tag_index = part_of_speech_index[nltk.pos_tag([dd[0]])[0][1]]
        tag_indicator_list.append(tag_index)
        data_ = dd[1].split(' ')
        data = []
        for p in data_: # delete stress numbers in the data
            if ('0' in p) or ('1' in p) or ('2' in p):
                data.append(p[:-1])
            else:
                data.append(p)
        data_dict[dd[0]] = data
        data_dict_with_stress_number[dd[0]] = data_
    return data_dict, data_dict_with_stress_number, tag_indicator_list

def making_target_label(data_dict_with_stress_number):
    v_phonemes_index = {'AA':0, 'AE':1, 'AH':2, 'AO':3, 'AW':4, 'AY':5, 'EH':6, 'ER':7,'EY':8, 'IH':9, 'IY':10, 'OW':11, 'OY':12, 'UH':13, 'UW':14}
    target = []
    for i in data_dict_with_stress_number:
        pos = 0
        for v in data_dict_with_stress_number[i]:
            if '1' in v:
                pos += 1
                target.append(pos)
                break
            elif ('0' in v) or ('2' in v):
                pos += 1
    return target

def vowel_consonant_sequence(data_dict):
    v_phonemes_index = {'AA':0, 'AE':1, 'AH':2, 'AO':3, 'AW':4, 'AY':5, 'EH':6, 'ER':7, 'EY':8, 'IH':9, 'IY':10, 'OW':11, 'OY':12, 'UH':13, 'UW':14}
    #c_phonemes_index = {'P':16, 'B':17, 'CH':18, 'D':19, 'DH':20, 'F':21, 'G':22, 'HH':23\
    #                    , 'JH':24, 'K':25, 'L':26, 'M':27, 'N':28, 'NG':29, 'R':30, \
    #                    'S':31, 'SH':32, 'T':33, 'TH':34, 'V':35, 'W':36, 'Y':37, 'Z':38, 'ZH':39}
    sequence_list = []
    for d in data_dict:
        sequence = str()
        for p in data_dict[d]:
            if p in v_phonemes_index:
                sequence += 'V'
            else:
                sequence += 'C'
        while (('CC' in sequence) or ('VV' in sequence)):
            sequence = sequence.replace('CC', 'C')
            sequence = sequence.replace('VV', 'V')
        sequence_list.append(sequence)
    return sequence_list


#
# actually this funct is to find:
# 1. the consonant phoneme before the first vowel phoneme
# 2. all the vowel phonemes, no 4 vowel phonemes, leave the place and mark it as "15"
# 3. the phoneme before the last vowel phoneme (and it is not necessary a consonant phoneme)
# 4. all phonemes transfer to dummy indexes, "15" means no such a phoneme
# 5. the last phoneme in the word, which may be the same as the last vowel phoneme
#
def vowels(data_dict):
    v_phonemes_index = {'AA':0, 'AE':1, 'AH':2, 'AO':3, 'AW':4, 'AY':5, 'EH':6, 'ER':7, 'EY':8, 'IH':9, 'IY':10, 'OW':11, 'OY':12, 'UH':13, 'UW':14}
    c_phonemes_index = {'P':16, 'B':17, 'CH':18, 'D':19, 'DH':20, 'F':21, 'G':22, 'HH':23, 'JH':24, 'K':25, 'L':26, 'M':27, 'N':28, 'NG':29, 'R':30, 'S':31, 'SH':32, 'T':33, 'TH':34, 'V':35, 'W':36, 'Y':37, 'Z':38, 'ZH':39}
    vowel_list = []
    for d in data_dict:
        sing_word_vowel = []
        first_vowel = 1
        for i in range(len(data_dict[d])):
            if data_dict[d][i] in v_phonemes_index:
                # The next 7 lines below is to find the consonant phoneme before the first vowel phoneme.
                if first_vowel:
                    if i > 0 :
                        sing_word_vowel.append(c_phonemes_index[data_dict[d][i-1]])
                    else:
                        sing_word_vowel.append(15) # a dummy label for no such a consonant phoneme
                    first_vowel = 0
                sing_word_vowel.append(v_phonemes_index[data_dict[d][i]]) # append the consonant phoneme
        while len(sing_word_vowel) < 5: # 5 includes 1 consonant phoneme place, 4 vowel phoneme places
            sing_word_vowel.append(15) # a dummy label for no such a vowel
        # The next 8 lines is to find the phoneme before the last vowel phoneme
        reverser = data_dict[d][::-1] # reverse it to find the phoneme before the last vowel
        for i in range(len(reverser)):
            if reverser[i] in v_phonemes_index:
                if reverser[i+1] in v_phonemes_index:
                    sing_word_vowel.append(v_phonemes_index[reverser[i+1]])
                else:
                    sing_word_vowel.append(c_phonemes_index[reverser[i+1]])
                break
        # The next 4 lines is to find the last phoneme in a word
        if data_dict[d][-1] in v_phonemes_index:
            sing_word_vowel.append(v_phonemes_index[data_dict[d][-1]])
        else:
            sing_word_vowel.append(c_phonemes_index[data_dict[d][-1]])
        vowel_list.append(sing_word_vowel)
    return vowel_list

#a = {'cbaabc': ['N', 'AA', 'N', 'P', 'OY', 'Z', 'AH', 'N', 'AH', 'S']}
#b = vowels(a)
#print(b)

def label_it(data_list):
    enc_label = LabelEncoder()
    return enc_label.fit_transform(data_list)

def finally_arranging_data(raw_data):
    # part_of_speech_list has been labeled, and need no more
    data_dict,  data_dict_with_stress_number, part_of_speech_list = making_data_dic_N_part_of_speech(raw_data)
    pronunciation_sequence = label_it(vowel_consonant_sequence(data_dict))
    features = vowels(data_dict)
    for i in range(len(features)):
        features[i].extend([part_of_speech_list[i], pronunciation_sequence[i]])
    labels = np.array(making_target_label(data_dict_with_stress_number))
    features = np.array(features)
    return features, labels

def test_data_arranging(raw_data):
    # part_of_speech_list has been labeled, and need no more
    data_dict,  data_dict_with_stress_number, part_of_speech_list = making_data_dic_N_part_of_speech(raw_data)
    pronunciation_sequence = label_it(vowel_consonant_sequence(data_dict))
    features = vowels(data_dict)
    for i in range(len(features)):
        features[i].extend([part_of_speech_list[i], pronunciation_sequence[i]])
    features = np.array(features)
    return features
        
    
def train(data, classifier_file): # weight {1:0.7,2:0.5,3:1,4:2}
    features, labels = finally_arranging_data(data)
    enc_encoder = OneHotEncoder(handle_unknown='ignore')
    fit_features = enc_encoder.fit(features)
    features = fit_features.transform(features).toarray()
    
    clf = LogisticRegression(solver='newton-cg',multi_class='ovr', C=20.0, class_weight={1:0.7,2:0.5,3:1,4:2}).fit(features, labels)    #class_weight='balanced'
    with open(classifier_file, 'wb') as handle:
        pickle.dump(fit_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return
    
def test(data, classifier_file):
    with open(classifier_file, 'rb') as handle:
        fit_features = pickle.load(handle)
        clf = pickle.load(handle)
    test_data_features = test_data_arranging(data)
    test_data_features = fit_features.transform(test_data_features).toarray()
    result = clf.predict(test_data_features)
    return result


#test_data = helper.read_data('./asset/tiny_test.txt')
#prediction = test(test_data, classifier_path)
#print(prediction)    
#ground_truth = [1, 1, 2, 1]
#print(f1_score(ground_truth, prediction, average='macro'))




'''
# This is for a large test.
def test1(data, classifier_file):
    with open(classifier_file, 'rb') as handle:
        fit_features = pickle.load(handle)
        clf = pickle.load(handle)
    features, labels = finally_arranging_data(data)
    features = fit_features.transform(features).toarray()
    result = clf.predict(features)
    return f1_score(result, labels, average='macro')

test_words = helper.read_data('./asset/test_words.txt')
print(test1(test_words, classifier_path))
'''
