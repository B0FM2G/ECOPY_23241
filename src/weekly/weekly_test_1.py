#B0FM2G
def evens_from_list(input_list):
    paros=[]
    for index, element in enumerate(input_list):
        if element%2==0:
            paros.append(element)
    return(paros)

def every_element_is_odd(input_list):
    for element in input_list:
        if element % 2 == 0:
            return False
    return True

def kth_largest_in_list(input_list,kth_largest):
    input_list = sorted(input_list)
    return(input_list[-kth_largest])

def cumavg_list(input_list):
    if not input_list:
        return []
    cumulative_sum = 0
    cumulative_averages = []
    for i, num in enumerate(input_list):
        cumulative_sum += num
        average = cumulative_sum / (i + 1)
        cumulative_averages.append(average)
    return cumulative_averages

def element_wise_multiplication(input_list1, input_list2):
    szorzat = []
    for index, element in enumerate(input_list1):
        szorzat.append(input_list1[index]*input_list2[index])
    return szorzat

def merge_lists(*lists):
    merged = []
    for i in lists:
        merged.extend(i)
    return merged

def squared_odds(input_list):
    paratlan_negyzet = []
    for index, element in enumerate(input_list):
        if element % 2 == 1:
            paratlan_negyzet.append(element**2)
    return(paratlan_negyzet)

def reverse_sort_by_key(input_dict):
    sorted_items = sorted(input_dict.items(), key=lambda x: x[0], reverse=True)
    reversed_dict = dict(sorted_items)
    return reversed_dict

def sort_list_by_divisibility(input_list):
    by_two=[]
    by_five=[]
    by_two_and_five=[]
    by_none=[]
    for element in input_list:
        if element%2==0 and element%5==0:
            by_two_and_five.append(element)
        elif element%5==0:
            by_five.append(element)
        elif element%2==0:
            by_two.append(element)
        else:
            by_none.append(element)
    div={'by_two':sorted(by_two), 'by_five':sorted(by_five), 'by_two_and_five':sorted(by_two_and_five), 'by_none':sorted(by_none)}
    return(div)