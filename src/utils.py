def read_file(file_name):
    with open(file_name) as f:
        return f.read()


def print_items(items):
    for item in items:
        print(item)
    print()


def get_n_sorted(n, items, key=None, get_max=False):
    return sorted(items, key=key, reverse=get_max)[:n]
