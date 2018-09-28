import sys

try:
    arity = input('Input an arity : ')
    if int(arity) < 0 or not arity.isdigit():
        raise ValueError
except ValueError:
    print('Incorrect arity, giving up...')
    sys.exit()
   
print('A term should contain only letters, underscores, commas, parentheses, spaces.')
im_input = input('Input a term: ')
im_input = im_input.replace(' ', '')
im_input = im_input.replace('_', 'f')
# the situation when arity is 0
if arity == '0' and im_input.isalpha():
    print('Good, the term is syntactically correct.')
    sys.exit()
elif im_input.isalpha():
    print('Unfortunately, the term is syntactically incorrect.')
    sys.exit()
# to transform the term into a "general status"
nn = 1
ff = 'f' + 'f' * nn
for i in im_input:
    if i.isalpha():
        im_input = im_input.replace(i, 'f')
while len(ff) < len(im_input):
    ff = 'f' + 'f' * nn
    while ff in im_input:
        im_input = im_input.replace(ff, 'f')
    nn += 1    
# to create a function template
f = 'f,' * int(arity)
f = f[ :-1]
fx = 'f(' + f + ')'
# to match the pattern
while fx in im_input:
    im_input = im_input.replace(fx, 'f')
# output
if im_input == 'f':
    print('Good, the term is syntactically correct.')
else:
    print('Unfortunately, the term is syntactically incorrect.')

