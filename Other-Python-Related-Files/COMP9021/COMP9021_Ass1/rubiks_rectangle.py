import sys
import copy
import itertools
from collections import deque
import math

def right_shift(que1, que2):
    que1 = deque(que1)
    que2 = deque(que2)
    rr1 = copy.deepcopy(que1)
    rr2 = copy.deepcopy(que2)
    p1 = rr1.pop()
    p2 = rr2.pop()
    rr1.appendleft(p1)
    rr2.appendleft(p2)
    return (list(rr1), list(rr2))

def row_exchange(que_1, que_2):
    que_3 = que_1
    que_1 = que_2
    que_2 = que_3
    return (que_1, que_2)

def middle_clockwise_rotation(que1_, que2_):
    que3_ = [que1_[0], que2_[1], que1_[1], que1_[3]]
    que4_ = [que2_[0], que2_[2], que1_[2], que2_[3]]
    return (que3_, que4_)


def cantor(qu_e1, qu_e2):
    qqu_e2 = copy.deepcopy(qu_e2)
    qqu_e2.reverse()
    c_list = qu_e1 + qqu_e2
    num = 0
    for i in range(0, 7):
        tmp = 0
        for j in range(i, 8):
            if c_list[i] > c_list[j]:
                tmp += 1
        num += tmp * math.factorial(7-i)
    return (num)

initial_conf = (['1', '2', '3', '4'], ['8', '7', '6', '5'])
ipt_strs = input('Input final configuration : ')
ipt_strs = ipt_strs.replace(' ', '')

if set(ipt_strs) != {'1', '2', '3', '4', '5', '6', '7', '8'}:
    print('Incorrect configuration , giving up...')
    sys.exit()

ipt_strs_tuple = (list(ipt_strs)[: 4], list(ipt_strs)[: 3: -1])
target_value = cantor(ipt_strs_tuple[0], ipt_strs_tuple[1])



queue = deque([initial_conf])
old_cantor = set()
n = 0
next_floor = []
while 1:
    i = queue.popleft()
    i_cantor = cantor(i[0], i[1])
    if i_cantor != target_value:
        if i_cantor not in old_cantor:
            old_cantor.add(i_cantor)
            next_floor.extend([right_shift(i[0], i[1]), middle_clockwise_rotation(i[0], i[1]), row_exchange(i[0], i[1])])
    else:
        print('{} steps are needed to reach the final configuration.'.format(n))
        sys.exit()
    if not len(queue) > 0:
        n += 1
        queue.extend(next_floor)


