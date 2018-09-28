# Implements a function to encode a single strictly positive natural number that in base 2,
# reads as b_1 ... b_n, as b_1b_1 ... b_nb_n, and to encode a sequence of strictly positive
# natural numbers N_1 ... N_k with k >= 2 as N_1* 0 ... 0 N_k* where for all 0 < i <= k,
# N_i* is the encoding of N_i.
# Implements a function to decode a natural number N into a sequence of (one or more)
# strictly positive natural numbers according to the previous encoding scheme,
# or return None in case N does not encode such a sequence.
# Prompts the user to enter a strictly positive natural number, applies the decoding function,
# and prints out base 2 representations of both the encoding number and the decoded sequence
# for verification purposes.
#
# Written by *** and Eric Martin for COMP9021

import sys


def encode(*integers):
    return int('0'.join(''.join(c + c for c in bin(i)[2: ]) for i in integers), 2)
    

def decode(integer):
    answer_list = []
    j = str()
    str1 = bin(integer)[2: ]
    while len(str1) > 1:
        if str1[0] == str1[1]:
            j += str1[0]
            str1 = str1[2: ]
        elif str1[0] == '0' and str1[1] == '1':
            j += ','
            str1 = str1[1: ]
        elif str1[0] == '1' and str1[1] == '0':
            return None  
    if len(str1) == 1:
        return None 
    if ',' in j:
        j_list = j.split(',')
        for nums in j_list:
            answer_list.append(int(nums, 2))
    else:
        answer_list = [int(j, 2)]
    return answer_list
integer = input('Input a strictly positive integer: ')
if integer[0] == '0' or not integer.isdigit():
    print('Incorrect input, giving up!')
    sys.exit()
integer = int(integer)
decoding = decode(integer)
if decoding is None:
    print('Incorrect encoding!')
else:
    print('This encodes: ', decoding)
    print('Checking: ')
    print('  In base 2, {} is {}'.format(integer, bin(integer)[2: ]))
    print('  In base 2, {} is: [{}]'.format(decoding, ', '.join(bin(i)[2: ] for i in decoding)))

