import sys
import copy
file_set = set()
result_dict = dict()
result_scores = int()
the_file = open('wordsEn.txt', 'r')
the_list = the_file.readlines()
the_file.close()
for i in range(len(the_list)):
    the_list[i] = the_list[i].rstrip()

l_scores = {'a': 2, 'b': 5, 'c': 4, 'd': 4, 'e': 1, 'f': 6, 'g': 5, 'h': 5, 'i': 1, 'j': 7, 'k': 6, 'l': 3, 'm': 5, 'n': 2, 'o': 3, 'p': 5, 'q': 7, 'r': 2, 's': 1, 't': 2, 'u': 4, 'v': 6, 'w': 6, 'x':7, 'y': 5, 'z': 7}
the_input = input("Enter between 3 and 10 lowercase letters : ")
the_input = the_input.replace(' ', '')
if not (the_input.isalpha() and the_input.islower() and len(the_input) > 2 and len(the_input) < 11):
    print('Incorrect input, giving up...')
    sys.exit()

input_set = set(the_input)
for row in the_list:
    file_set = set(row)
    if ((input_set & file_set) == file_set): #and (file_set & set('aeiou') != set()):#(len(file_set) == len(row)) and 
        row_list = list(row)
        for character in the_input:
            if character in row_list:
                row_list.remove(character)
        if not row_list == []:
            continue
        for letter in row:
            if not letter == "'":
                result_scores += l_scores[letter]
        if str(result_scores) in result_dict.keys():
            result_dict[str(result_scores)].append(row)
        else:
            result_dict[str(result_scores)] = [row]
        result_scores = 0
if len(result_dict.keys()) == 0:
    print('No word is built from some of those letters.')
    
elif len(result_dict.keys()) == 1:
    for only_1 in result_dict:
        print('The highest score is', only_1)
        if len(result_dict[only_1]) == 1:
            print('The highest scoring word is', *result_dict[only_1])
        else:
            print('The highest scoring words are, in alphabetical order :')
            for print_results in result_dict[only_1]:
                print('    ',print_results)
else:
    result_keys = list(map(int, result_dict))
    result_keys.sort(reverse = True)
    print('The highest score is', result_keys[0])
    if len(result_dict[str(result_keys[0])]) == 1:
        print('The highest scoring word is', *result_dict[str(result_keys[0])])
    else:
        print('The highest scoring words are , in alphabetical order :')
        for print_results1 in result_dict[str(result_keys[0])]:
                print('    ', print_results1)


