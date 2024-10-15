import re
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <gprof_output.txt>")
    sys.exit(1)

gprof_output_file = sys.argv[1]
call_stacks = []

with open(gprof_output_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('[') and ']' in line:
            call_stack = re.findall(r'\[(.*?)\]', line)
            call_stacks.append(';'.join(call_stack))

with open('gprof_stacks.txt', 'w') as f:
    for stack in call_stacks:
        f.write(f"{stack} 1\n")
