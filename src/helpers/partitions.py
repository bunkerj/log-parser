from src.helpers.partition_item import ParititionItem


class Partitions:
    def __init__(self):
        self.partition_list = []

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self.partition_list):
            result = self.partition_list[self.idx]
            self.idx += 1
            return result
        else:
            raise StopIteration

    def add(self, partition, step_num):
        if len(partition) == 0:
            raise Exception('Partition cannot be empty')
        partition_item = ParititionItem(partition, step_num)
        self.partition_list.append(partition_item)
