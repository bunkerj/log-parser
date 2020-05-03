import os
import csv
from glob import glob
from src.utils import read_csv


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
        indices = set(idx + 1 for idx in indices)
        indices.add(0)
    with open(output_path, 'w+', encoding='utf-8') as output_file:
        with open(input_path, 'r+', encoding='utf-8') as input_file:
            for idx, line in enumerate(input_file.readlines()):
                if idx in indices:
                    output_file.write(line)


def get_num_lines(file_path):
    return sum(1 for _ in open(file_path, 'r', encoding='utf-8'))


def get_nested_file_paths(input_dir, extension):
    return [y for x in os.walk(input_dir) for y in
            glob(os.path.join(x[0], extension))]


def extract_templates(structured_file):
    templates = {}
    lines = read_csv(structured_file)
    for idx in range(1, len(lines)):
        event_id = int(lines[idx][-2][1:])
        template = lines[idx][-1]
        templates[event_id] = template
    return {k: templates[k] for k in sorted(templates)}
