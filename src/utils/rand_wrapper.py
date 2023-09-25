import random
def random_from_list(input_list):
    veletlen=random.choice(input_list)
    return veletlen

def random_sublist_from_list(input_list, number_of_elements):
    return random.choices(input_list,weights=None, cum_weights=None, k=number_of_elements)

def random_from_string(input_string):
    return random.choice(input_string)

def hundred_small_random():
    veletlen=list([random.random() for _ in range(100)])
    return veletlen

def hundred_large_random():
    veletlen=list([random.randint(10,1000) for _ in range(100)])
    return veletlen

def five_random_number_div_three():
    by_three=[]
    while len(by_three) <5:
        elem=random.randrange(9, 999, 3)
        by_three.append(elem)
    return by_three

def random_reorder(input_list):
    veletlen=random.sample(input_list, k=len(input_list))
    return veletlen

def uniform_one_to_five():
    return random.uniform(1,6)