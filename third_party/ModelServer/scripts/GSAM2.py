#!/usr/bin/env python3
"""
服务器启动器
"""
import argparse
from ModelServer import start_server
from ModelServer.models.GSAM2.GSAM2 import GSAM2


# 使用示例
if __name__ == "__main__":        
    args = argparse.ArgumentParser()
    args.add_argument('--port', type=int, default=7000)
    args = args.parse_args()
    start_server(GSAM2(), port=args.port)