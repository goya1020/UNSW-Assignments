import sys
import copy
file_set = set()
result_dict = dict()
result_scores = int()
im_input = input('Which data file do you want to use? ')
the_file = open(im_input, 'r')
the_list = the_file.readlines()
the_file.close()
for p in range(len(the_list)):
    the_list[p] = the_list[p].strip('\n') 
    
remove_list = []
lenth = 0
for i in the_list:
    x1 = int(i.split(' ')[0])
    y1 = int(i.split(' ')[1])
    x2 = int(i.split(' ')[2])
    y2 = int(i.split(' ')[3])
    for ii in the_list[the_list.index(i)+1: ]:
        x11 = int(ii.split(' ')[0])
        y11 = int(ii.split(' ')[1])
        x22 = int(ii.split(' ')[2])
        y22 = int(ii.split(' ')[3])
        if (x1 < x11) and (y1 < y11) and (x2 > x22) and (y2 > y22):
            remove_list.append(ii)
        elif (x1 > x11) and (y1 > y11) and (x2 < x22) and (y2 < y22):
            remove_list.append(i)
            
for remove in remove_list:
    if remove in the_list:
        the_list.remove(remove)
        
lenth = 0
for x in the_list:
    current_overlap = list()
    x_1 = int(x.split(' ')[0])
    y_1 = int(x.split(' ')[1])
    x_2 = int(x.split(' ')[2])
    y_2 = int(x.split(' ')[3])
    xi = x_1
    yi = y_1
    Mid = False
    for xx in the_list:
        x_11 = int(xx.split(' ')[0])
        y_11 = int(xx.split(' ')[1])
        x_22 = int(xx.split(' ')[2])
        y_22 = int(xx.split(' ')[3])
        if xx == x:
            continue
        if not (x_11 > x_2 or y_1 > y_22 or x_1 > x_22 or y_11 > y_2):
            current_overlap.append(xx)
    k = 1
    while 1:
        if Mid == False:
            if yi != y_2:
                yi += 1
                k = 1
            elif xi != x_2:
                xi += 1
                k = 2
            else:
                yi -= 1
                Mid = True
        else:
            if yi != y_1:
                yi -= 1
                k = 3
            elif xi != x_1:
                xi -= 1
                k = 4
            else:
                break
        for xxx in current_overlap:
            xo1 = int(xxx.split(' ')[0])
            yo1 = int(xxx.split(' ')[1])
            xo2 = int(xxx.split(' ')[2])
            yo2 = int(xxx.split(' ')[3])
            if ((xo1 <= xi) and (xo2 >= xi) and (yo1 <= yi) and (yo2 >= yi)):
                if k == 1 and yi == yo1:
                    lenth += 1
                elif k == 2 and xi == xo1:
                    lenth += 1
                elif k == 3 and yi == yo2:
                    lenth += 1
                elif k == 4 and xi == xo2:
                    lenth += 1
                break
        else:
            lenth += 1
            
print("The perimeter is:",int(lenth))    

