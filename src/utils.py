import csv


def read_file(file_path):
    with open(file_path) as f:
        return f.read()


def read_csv(file_path):
    with open(file_path) as f:
        csv_reader = csv.reader(f, delimiter=',')
        return list(csv_reader)


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
