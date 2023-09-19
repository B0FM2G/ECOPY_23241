def contains_value(input_list, element):
    return(element in input_list)

def number_of_elements_in_list(input_list):
    return(len(input_list))

def remove_every_element_from_list(input_list):
    del(input_list[:])
    return(input_list)

def reverse_list(input_list):
    input_list.reverse()
    return(input_list)

def odds_from_list(input_list):
    paratlan=[]
    for index, element in enumerate(input_list):
        if element%2==1:
            paratlan.append(element)
    return(paratlan)

def number_of_odds_in_list(input_list):
    return(len(odds_from_list(input_list)))

def contains_odd(input_list):
    for element in input_list:
        if element%2==1:
            return(True)
    return(False)

def contains_odd2(input_list):
    return(number_of_odds_in_list(input_list)>0)

def second_largest_in_list(input_list):
    input_list = sorted(input_list)
    return(input_list[-2])

def sum_of_elements_in_list(input_list):
    return(float(sum(input_list)))

def cumsum_list(input_list):
    kumulalt = []
    valtozo = 0
    for element in input_list:
        valtozo=element+valtozo
        kumulalt.append(valtozo)
    return(kumulalt)

def element_wise_sum(input_list1, input_list2):
    osszeg=[]
    for index,element in enumerate(input_list1):
        osszeg.append(input_list1[index]+input_list2[index])
    return(osszeg)

def subset_of_list(input_list,start_index,end_index):
    resz_lista=input_list[start_index:end_index+1]
    return(resz_lista)

def every_nth(input_list, step_size):
    return(input_list[::step_size])

def only_unique_in_list(input_list):
    return(len(set(input_list)) == len(input_list))

def keep_unique(input_list):
    input_list = list(set(input_list))
    return(input_list)

def swap(input_list,first_index,second_index):
    input_list[first_index], input_list[second_index] = input_list[second_index], input_list[first_index]
    return(input_list)

def remove_element_by_value(input_list, value_to_remove):
    input_list.remove(value_to_remove)
    return(input_list)

def remove_element_by_index(input_list, index):
    del(input_list[index])
    return(input_list)

def multiply_every_element(input_list, multiplier):
    for i, index in enumerate(input_list):
        input_list[i]=input_list[i]*multiplier
    return(input_list)

def remove_key(input_dict, key):
    if key in input_dict:
        input_dict.pop(key)
    return(input_dict)

def sort_by_key(input_dict):
    myKeys = list(input_dict.keys())
    myKeys.sort()
    input_dict = {i: input_dict[i] for i in myKeys}
    return(input_dict)

def sum_in_dict(input_dict):
    osszeg=sum(input_dict.values())
    return(osszeg)

def merge_two_dicts(input_dict1, input_dict2):
    input_dict2.update(input_dict1)
    return(input_dict2)

def merge_dicts(*dicts):
    merged={}
    for c in dicts:
        merged.update(c)
    return(merged)

def sort_list_by_parity(input_list):
    paratlan=[]
    paros=[]
    for element in input_list:
        if element%2==1:
            paratlan.append(element)
        else:
            paros.append(element)
    parity={'even':paros, 'odd':paratlan}
    return(parity)

def mean_by_key_value(input_dict):
    new_dict={}
    for key, value in input_dict.items():
        new_dict[key]=sum(value)/len(value)
    return(new_dict)

def count_frequency(input_list):
    freq_dict = {}
    for element in input_list:
        freq_dict[element] = input_list.count(element)
    return(freq_dict)