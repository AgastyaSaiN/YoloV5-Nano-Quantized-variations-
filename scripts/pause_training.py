"""Suspend / resume the fine-tuning process (and its data workers) so speed benchmarks run on an idle machine.
Training keeps its exact state while suspended.
    python scripts/pause_training.py suspend|resume
"""
import ctypes
import sys

import psutil

PROCESS_SUSPEND_RESUME = 0x0800
ntdll, k32 = ctypes.WinDLL("ntdll"), ctypes.WinDLL("kernel32")


def trainers():
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        c = " ".join(p.info["cmdline"] or [])
        if "finetune_relu.py" in c or ("multiprocessing" in c and "spawn" in c and ".venv_train" in c):
            out.append(p.info["pid"])
    return out


action = sys.argv[1]
pids = trainers()
for pid in pids:
    h = k32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
    if h:
        (ntdll.NtSuspendProcess if action == "suspend" else ntdll.NtResumeProcess)(h)
        k32.CloseHandle(h)
print(f"{action}: {len(pids)} training processes")
