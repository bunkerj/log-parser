"""
Example:
py split_into_assign_and_unstruct TARGET ASSIGN_PATH UNSTRUCT_PATH
"""
import sys
import pandas as pd

TARGET = sys.argv[1]
ASSIGN_PATH = sys.argv[2]
UNSTRUCT_PATH = sys.argv[3]

df = pd.read_csv(TARGET, header=0, names=['LineId', 'Content', 'EventId',
                                          'EventTemplate', 'Params'])
df.to_csv(ASSIGN_PATH, columns=['LineId', 'EventId'], index=False)
df.to_csv(UNSTRUCT_PATH, columns=['Content'], index=False, header=False)
