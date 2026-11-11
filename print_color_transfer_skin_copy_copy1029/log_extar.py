#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
过滤日志中的 tqdm 进度条行，其余内容全部输出。
用法: python filter_tqdm.py [日志文件]
     或 cat 日志 | python filter_tqdm.py
"""

import re
import sys

def is_tqdm_line(line: str) -> bool:
    """
    判断一行是否为 tqdm 进度条行。
    特征：包含进度条符号（如 █ 或 |）且以“Epoch 数字/数字:”开头，
    或包含“%|”模式。
    """
    # 如果行以 "Epoch" 开头，且包含进度条常见符号
    if re.match(r'Epoch\s+\d+/\d+:', line):
        # 检查是否包含进度条字符（如 █, ▉, | 等）
        if re.search(r'[█▉▊▋▌▍▎▏│]', line) and '%' in line:
            return True
        # 或者包含百分比与进度条格式
        if re.search(r'\d+%\|[ \▉▊▋▌▍▎▏█-]*\|', line):
            return True
    # 通用的 tqdm 特征：包含 "%|" 和 "|" 并且有进度条符号
    if re.search(r'%\|[ \▉▊▋▌▍▎▏█-]+\|', line):
        return True
    return False

def main():
    # 读取输入
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding='utf-8') as f:
            lines = f.read().splitlines()
    else:
        lines = sys.stdin.read().splitlines()

    # 输出所有非 tqdm 行
    for line in lines:
        if not is_tqdm_line(line):
            print(line)

if __name__ == '__main__':
    main()