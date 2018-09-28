import sys
remainder = int()
i = 2
the_result = str()
nn_integer = input('Input a nonnegative integer : ')
if nn_integer[0] == '0' or not nn_integer.isdigit():
    print('Incorrect input, giving up...')
    sys.exit()
nn_integer = int(nn_integer)
the_original_number = nn_integer
while nn_integer:
    num_storage = nn_integer
    nn_integer = nn_integer // i
    remainder = num_storage % i
    the_result += str(remainder)
    i += 1
print('Decimal {} reads as {} in factorial base.'.format(the_original_number, the_result[:: -1]))

