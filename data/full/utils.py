import csv


def read_template_assignments_from_file(file_path, jump_size=1):
    lines = []
    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for idx, line in enumerate(csv_reader):
            if idx != 0 and (idx - 1) % jump_size == 0:
                lines.append(line[-1].strip())
    return lines


def subsample_file(input_path, output_path, indices, include_header=False):
    if include_header:
        indices = [idx + 1 for idx in indices]
        indices.insert(0, 0)
    with open(output_path, 'w+', encoding='utf-8') as output_file:
        with open(input_path, 'r+', encoding='utf-8') as input_file:
            for idx, line in enumerate(input_file.readlines()):
                if idx in indices:
                    output_file.write(line)


def get_file_length(file_path):
    return sum(1 for i in open(file_path, 'r', encoding='utf-8'))
