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
