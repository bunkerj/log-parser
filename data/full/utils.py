def readlines_with_jump(file_path, jump_size):
    lines = []
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if idx % jump_size == 0:
                lines.append(line.strip())
    return lines
