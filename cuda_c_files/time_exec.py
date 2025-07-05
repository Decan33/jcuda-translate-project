import subprocess
import time
import statistics
import sys
import os

CUDA_FILES = [
    "idea_cuda_pinned.cu",
    "idea_cuda.cu",
    "idea_cuda_raw.cu",
    "idea_cuda_raw_improved.cu",
    "idea_cuda_stream_async.cu",
    "idea_cuda_stream.cu"
]

CUDA_DIR = os.path.dirname(os.path.abspath(__file__))


def compile_cuda_file(cuda_file):
    exe_name = os.path.splitext(cuda_file)[0] + ".exe"
    print(exe_name)
    cuda_path = os.path.join(CUDA_DIR, cuda_file)
    print(cuda_path)
    exe_path = os.path.join(CUDA_DIR, exe_name)
    print(exe_path)
    bin_path = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.41.34120\\bin"
    print(f"Compiling {cuda_file}...")
    try:
        subprocess.run([
            "nvcc", "-ccbin", bin_path, cuda_path, "-o", exe_path, "-I", CUDA_DIR
        ], check=True)
        print(f"Compiled {cuda_file} to {exe_name}")
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed for {cuda_file}: {e}")
        return None
    return exe_path

def time_executable(executable_path, args=None):
    print(f"Running {executable_path}...")
    try:
        subprocess.run([executable_path] + (args or []), check=True)
    except subprocess.CalledProcessError as e:
        print(f"Execution failed: {e}")


def main():
    for cuda_file in CUDA_FILES:
        print(f"\n=== Processing {cuda_file} ===")
        exe_path = compile_cuda_file(cuda_file)
        if exe_path is None:
            continue
        print(f"Currently running: {os.path.basename(exe_path)}")
        time_executable(exe_path) # Actual test

if __name__ == "__main__":
    main()
