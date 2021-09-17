import numpy as np

#count how many times an element occurs in an array
def count(elem, array):
    ct = 0
    for element in array:
        if elem == element:
            ct+=1
    return ct

#calculate the frequency of numbers in the set
def freq(x):
    # freqs = [(value, count(value, x) / len(x)) for value in set(x)] 
    freqs = [[value, count(value, x)] for value in set(x)] 
    return freqs




