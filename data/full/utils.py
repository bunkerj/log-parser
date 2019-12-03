import csv


def read_template_assignments_from_file(file_path, jump_size):
    lines = []
    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for idx, line in enumerate(csv_reader):
            if idx != 0 and (idx - 1) % jump_size == 0:
                lines.append(line[-1].strip())
    return lines
