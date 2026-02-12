#!/usr/bin/env python3
import sys
import re

def filter_cuda(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read()
            
            # 1. Remove the CUDA launch configuration <<<...>>>
            # This regex handles multi-line launches (re.DOTALL)
            # It replaces kernel<<<...>>>(args) with kernel(args)
            content = re.sub(r'<<<.*?>>>', '', content, flags=re.DOTALL)
            
            # 2. Output the "cleaned" code to stdout for Doxygen
            print(content)
    except Exception as e:
        # If something fails, just print the original content so we don't break everything
        print(content)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filter_cuda(sys.argv[1])
