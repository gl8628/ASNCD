# -*- coding: utf8 -*-
import os
import re
import shutil
import subprocess
import time



xmeasures_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "xmeasures-master", "bin", "Release", "xmeasures")

tmp_dir = os.path.join(os.environ.get("TEMP") or "/tmp", "xmeasures")


def f1(pred, target,dataname):
    return run_xmeasures("-f", pred, target,dataname)


def nmi(pred, target,dataname):
    return run_xmeasures("-n", pred, target,dataname)

def omega(pred, target,dataname):
    return run_xmeasures("-o", pred, target,dataname)

def run_xmeasures(args, pred, target,dataname):
    # 节点集合取 pred 与 target 的并集，避免 KeyError（pred 中可能含有
    # target 未覆盖的背景节点）。xmeasures 要求两边的节点都是 0..N-1 的整数，
    # 因此统一重映射到并集的连续索引即可。
    nodes = set(n for c in target for n in c) | set(n for c in pred for n in c)
    node_idx = {n: i for i, n in enumerate(sorted(nodes))}

    pred = [[node_idx[n] for n in c] for c in pred]
    target = [[node_idx[n] for n in c] for c in target]

    # 每个进程 / 每次调用使用独立目录（PID + 纳秒时间戳），避免并发评估
    # 同名数据集时互相删除或覆盖 pred.cnl / target.cnl。
    safe_name = re.sub(r"[^\w.\-]+", "_", str(dataname or "unknown")).strip("_")
    path = os.path.join(
        tmp_dir,
        (safe_name[:64] or "unknown") + f"_{os.getpid()}_{time.time_ns()}")
    os.makedirs(path, exist_ok=True)

    pred_path = os.path.join(path, "pred.cnl")
    target_path = os.path.join(path, "target.cnl")

    def _write_cnl(file_path, comms):
        with open(file_path, 'w', encoding='utf8') as f:
            for c in comms:
                f.write(" ".join([str(n) for n in c]))
                f.write("\n")
            f.flush()
            os.fsync(f.fileno())  # 落盘后再交给外部进程读取

    try:
        _write_cnl(pred_path, pred)
        _write_cnl(target_path, target)

        cmd_onmi = [
            xmeasures_bin,
            args,
            pred_path,
            target_path
        ]

        out = subprocess.run(cmd_onmi, capture_output=True)
        if out.returncode != 0:
            raise RuntimeError(out.stderr)

        out = out.stdout.decode("utf-8")

        res = out.split("\n")[-2]
        return res
    finally:
        shutil.rmtree(path, ignore_errors=True)


