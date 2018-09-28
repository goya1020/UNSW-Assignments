# Uses Heath Nutrition and Population statistics, avalaible at
# http://datacatalog.worldbank.org, stored in the file HNP_Data.csv,
# assumed to be stored in the working directory.
# Prompts the user for an Indicator Name. If it exists and is associated with
# a numerical value for some countries or categories, for some the years 1960-2015,
# then finds out the maximum value, and outputs:
# - that value;
# - the years when that value was reached, from oldest to more recents years;
# - for each such year, the countries or categories for which that value was reached,
#   listed in lexicographic order.
# 
# Written by *** and Eric Martin for COMP9021

import sys
import os
import csv

def if_a_number(number):
    try:
        number = float(number)
    except ValueError:
        return False
    else:
        return True

filename = 'HNP_Data.csv'
if not os.path.exists(filename):
    print('There is no file named {} in the working directory, giving up...'.format(filename))
    sys.exit()

indicator_of_interest = input('Enter an Indicator Name: ')

first_year = 1960
number_of_years = 56
max_value = None
countries_for_max_value_per_year = {}
has_i_n = []
best_value = 0
has_best_value = []

with open(filename) as csvfile:
    for line in csvfile:
        if indicator_of_interest in line:
            has_i_n.append(line)
    for information in has_i_n:
        information_set = set(information.split(','))
        for something in information_set:
            if if_a_number(something):
                if eval(something) > best_value:
                    best_value = eval(something)                
    if not best_value == 0:
        max_value = best_value
    while has_i_n != []:
        pop_ = has_i_n.pop()
        if str(best_value) in pop_:
            has_best_value.append(pop_)
    
    for contents in has_best_value:
        n = 0
        if contents.find('"') == 0:
            country = contents[1 : contents.find('"', 1)]
            contents = contents.split('"')
            contents = contents[-1].split(',')[2: ]
            while contents != []:
                pop_contents = contents.pop()
                if pop_contents == str(max_value):
                    if 2015 - n in countries_for_max_value_per_year:
                        if country not in countries_for_max_value_per_year[2015 - n]: 
                            countries_for_max_value_per_year[2015 - n].append(country)
                    else:
                        countries_for_max_value_per_year[2015 - n] = [country]
                n += 1
        
        elif ',' in indicator_of_interest:
            contents = contents.replace('"' + indicator_of_interest + '"' + ',', '')
            country, cc, ic, values = contents.split(',', 3)
            values = values.split(',')
            while values != []:
                pop_values = values.pop()
                if pop_values == str(max_value):
                    if 2015 - n in countries_for_max_value_per_year:
                        if country not in countries_for_max_value_per_year[2015 - n]: 
                            countries_for_max_value_per_year[2015 - n].append(country)
                    else:
                        countries_for_max_value_per_year[2015 - n] = [country]
                n += 1

        
        else:
            country, cc, i_n, ic, values = contents.split(',', 4)
            values = values.split(',')
            while values != []:
                pop_values = values.pop()
                if pop_values == str(max_value):
                    if 2015 - n in countries_for_max_value_per_year:
                        if country not in countries_for_max_value_per_year[2015 - n]: 
                            countries_for_max_value_per_year[2015 - n].append(country)
                    else:
                        countries_for_max_value_per_year[2015 - n] = [country]
                n += 1
                
for a_year in countries_for_max_value_per_year:
    countries_for_max_value_per_year[a_year].sort()
        
if max_value == None:
    print('Sorry, either the indicator of interest does not exist or it has no data.')
else:
    print('The maximum value is:', max_value)
    print('It was reached in these years, for these countries or categories:')
    for year in sorted(countries_for_max_value_per_year):
        print('    {}: {}'.format(year, countries_for_max_value_per_year[year]))


