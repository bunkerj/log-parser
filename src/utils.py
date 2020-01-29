import csv
import numpy as np
from random import uniform
from scipy.cluster.hierarchy import dendrogram


def read_file(file_path):
    with open(file_path) as f:
        return f.read()


def read_csv(file_path):
    with open(file_path) as f:
        csv_reader = csv.reader(f, delimiter=',')
        return list(csv_reader)


def get_template_assignments(file_path):
    return read_csv(file_path)[1:]


def write_csv(file_path, content_dict):
    with open(file_path, mode='w+', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(content_dict.keys())
        content_list = get_content_list(content_dict)
        if len(content_list) != 0:
            for line_idx in range(len(content_list[0])):
                line_contents = []
                for key in content_dict:
                    line_contents.append(content_dict[key][line_idx])
                csv_writer.writerow(line_contents)


def get_content_list(content_dict):
    n_ref = None
    content_list = []
    for key in content_dict:
        n = len(content_dict[key])
        if n_ref is None:
            n_ref = n
        elif n_ref != n:
            error_msg = 'Invalid content length ---> Expected: {} / Actual: {}'
            raise Exception(error_msg.format(n_ref, n))
        content_list.append(content_dict[key])
    return content_list


def print_items(items):
    for item in items:
        print(item)
    print()


def get_n_sorted(n, items, key=None, get_max=False):
    return sorted(items, key=key, reverse=get_max)[:n]


def are_lists_equal(list1, list2):
    if len(list1) != len(list2):
        return False
    sorted_list1 = sorted(list1)
    sorted_list2 = sorted(list2)
    for idx in range(len(sorted_list1)):
        if sorted_list1[idx] != sorted_list2[idx]:
            return False
    return True


def delete_indices_from_list(base_list, indices):
    for idx in sorted(indices, reverse=True):
        del base_list[idx]


def has_digit(token):
    return any(char.isdigit() for char in token)


def get_random_parameter_tuple(parameter_ranges_dict):
    return tuple(uniform(*parameter_ranges_dict[parameter_field])
                 for parameter_field in parameter_ranges_dict)


def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)
