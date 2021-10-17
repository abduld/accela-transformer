#!/usr/bin/env python3
import re
import sys
import os

regex = r"### \[(.*)\]"

input_file = sys.argv[1]
input_file_basename = os.path.basename(input_file)


with open(input_file) as f:
    file_content = f.read()
    is_even = False
    
    matches = re.finditer(regex, file_content, re.MULTILINE)
    for matchNum, match in enumerate(matches, start=1):
        is_even = not is_even
        if is_even:
            continue
        print(f"[{input_file_basename}]({input_file_basename} ':include :type=code python :fragment={match[1]}')\n")

