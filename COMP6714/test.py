import submission as submission
from os import walk
import codecs


# Output file name to store the final trained embeddings.
model_file = 'adjective_embeddings.txt'
# model_file = 'word2vec.txt'
# Fixed parameters
nums = 100001
embedding_dim = 200

input_dir = './BBC_Data.zip'
data_file = submission.process_data(input_dir)
# data_file = './processed_file.pickle'

# Train Embeddings, and write embeddings in "adjective_embeddings.txt"
submission.adjective_embeddings(
    data_file, model_file, nums, embedding_dim)
print('traning steps are all done.')
adj = ['able', 'average', 'bad', 'best', 'big', 'certain', 'common', 'current', 'different', 'difficult', 'early', 'extra', 'fair', 'few', 'final', 'former', 'great', 'hard', 'high', 'huge', 'important', 'key', 'large', 'last', 'less', 'likely', 'little', 'major', 'more', 'most', 'much', 'new', 'next', 'old', 'prime', 'real', 'recent', 'same', 'serious', 'short', 'small', 'top', 'tough', 'wide']

# adj = ['new', 'more', 'last', 'best', 'next', 'many', 'good']

# for i in adj:

#     input_adjective = i # 'bad'
#     top_k = 10
#     b = submission.Compute_topk(model_file, input_adjective, top_k)
#     print('Synonyms of \'{}\' is {}'.format(i, b))
#     print()


##########################
f = []
evaluation = './dev_set/'
for (dirpath, dirnames, filenames) in walk(evaluation):
    f.extend(filenames)
    break


file_num = 0
accumulate_hits = 0
for i in f:
    file_dir = evaluation + str(i) 
    file_num += 1
    with open(file_dir) as ff:
        lines = codecs.open(file_dir, 'r', encoding='utf-8').readlines()
        lines = list(map(str.strip, lines))
    mine = submission.Compute_topk(model_file, i, 100)
    # mine1 = [x.encode('utf-8') for x in mine]
    samething = [l for l in mine if l in lines]
    accumulate_hits += len(samething)
    print(i)
    # print(mine)
    # print(lines)
    print (samething)
    print (len(samething))
    print()
print('file num is:', file_num)
print('ave hits are:', accumulate_hits/file_num)
############################