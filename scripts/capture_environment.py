"""Record the exact hardware / software the benchmarks ran on -> results/environment.json"""
import json
import platform
import subprocess
import sys
from pathlib import Path

DAY2 = Path(__file__).resolve().parent.parent


def ps(cmd):
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout.strip()


def pkgs(py):
    code = ("import importlib,json;o={};\nfor m in ['numpy','onnxruntime','onnx','tensorflow','openvino','torch','cv2','pycocotools']:\n"
            " try: x=importlib.import_module(m); o[m]=getattr(x,'__version__',None) or x.get_version()\n except Exception: pass\nprint(json.dumps(o))")
    r = subprocess.run([py, "-c", code], capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {}


import cpuinfo  # noqa: E402
flags = cpuinfo.get_cpu_info().get("flags", [])
env = {
    "cpu": ps("(Get-CimInstance Win32_Processor).Name"),
    "physical_cores": int(ps("(Get-CimInstance Win32_Processor).NumberOfCores")),
    "logical_processors": int(ps("(Get-CimInstance Win32_Processor).NumberOfLogicalProcessors")),
    "base_clock_mhz": int(ps("(Get-CimInstance Win32_Processor).MaxClockSpeed")),
    "isa_extensions_relevant_to_int8": [f for f in flags if f in ("avx2", "avx512f", "avx512bw", "avx512vl", "avx512vnni", "fma")],
    "ram_gb": round(float(ps("(Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize")) / 1048576, 1),
    "gpu": ps("(Get-CimInstance Win32_VideoController | Select -ExpandProperty Name) -join ' | '"),
    "os": ps("(Get-CimInstance Win32_OperatingSystem).Caption + ' ' + (Get-CimInstance Win32_OperatingSystem).Version"),
    "power_plan": ps("(powercfg /getactivescheme)"),
    "python": platform.python_version(),
    "packages_main_env": pkgs(sys.executable),
    "packages_tensorflow_env": pkgs(str(DAY2 / ".venv/Scripts/python.exe")),
    "benchmark_protocol": {
        "batch_size": 1, "input": "1x3x640x640 float32 (preprocessing timed separately)",
        "threads": 4, "thread_placement": "process pinned to logical CPUs 0,2,4,6 = four distinct physical cores (SMT siblings left idle)",
        "process_priority": "high", "warmup_calls_per_round": 20, "timed_calls_per_round": 100, "rounds": 9,
        "round_order": "all models interleaved, shuffled each round, fresh process per measurement",
        "statistic": "median of per-round medians; round-to-round spread reported"},
}
(DAY2 / "results" / "environment.json").write_text(json.dumps(env, indent=2))
print(json.dumps(env, indent=2))
