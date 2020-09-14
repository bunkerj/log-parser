"""
Example:
py split_into_assign_and_unstruct TARGET ASSIGN_PATH UNSTRUCT_PATH
"""

import sys
import pandas as pd

TARGET = sys.argv[1]
ASSIGN_PATH = sys.argv[2]
UNSTRUCT_PATH = sys.argv[3]

df = pd.read_csv(TARGET)

with open(ASSIGN_PATH, 'w+', encoding='utf-8') as f_write:
    f_write.write('{}\n'.format('LineId,EventTemplate'))
    for idx, match_idx in enumerate(df['match_id']):
        f_write.write('{},{}\n'.format(idx + 1, match_idx))

with open(UNSTRUCT_PATH, 'w+', encoding='utf-8') as f_write:
    for log_message in df['log_message']:
        f_write.write('{}\n'.format(log_message))
