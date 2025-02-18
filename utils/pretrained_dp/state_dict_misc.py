import copy


def state_dict_filter_and_generator(source_state_dict, filter_prefix, omit_flag=False):
    target_state_dict = {}
    for k, v in source_state_dict.items():
        if k.startswith(filter_prefix):
            new_k = k[len(filter_prefix) + 1:]
            target_state_dict[new_k] = v
    if omit_flag and (not target_state_dict):
        return source_state_dict
    else:
        return target_state_dict

def state_dict_prefix_adder(source_state_dict, prefix):
    target_state_dict = {}
    for k, v in source_state_dict.items():
        new_k = prefix + '.' + k
        target_state_dict[new_k] = v
    return target_state_dict

def state_dict_deleteor(source_state_dict, prefix):
    source_state_dict_to_del = copy.deepcopy(source_state_dict)
    for k, v in source_state_dict.items():
        if k.startswith(prefix):
            del source_state_dict_to_del[k]
    return source_state_dict_to_del

def state_dict_preffix_updator(source_state_dict, old_preffix, new_preffix):
    target_state_dict = {}
    for k, v in source_state_dict.items():
        if k.startswith(old_preffix):
            new_k = new_preffix + k[len(old_preffix):]
            target_state_dict[new_k] = v
        else:
            target_state_dict[k] = v
    return target_state_dict