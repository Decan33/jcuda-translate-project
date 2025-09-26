import subprocess
import sys
import os

CUDA_FILES = [
    "idea_cuda_raw.cu",
    "idea_cuda_optimized.cu",
    "idea_cuda_pinned.cu",
    "idea_cuda_stream.cu",
    "idea_cuda_streams_and_shared.cu",
    "cuda_c_memory_test.cu",
    ""
]

CUDA_DIR = os.path.dirname(os.path.abspath(__file__))


def compile_cuda_file(cuda_file):
    exe_name = os.path.splitext(cuda_file)[0] + ".exe"
    cuda_path = os.path.join(CUDA_DIR, cuda_file)
    exe_path = os.path.join(CUDA_DIR, exe_name)
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
    print(f"Running {executable_path} {' '.join(args or [])}...")
    try:
        subprocess.run([executable_path] + (args or []), check=True)
    except subprocess.CalledProcessError as e:
        print(f"Execution failed: {e}")


def main():
    for cuda_file in CUDA_FILES:
        if not cuda_file:
            continue

        print(f"\n=== Processing {cuda_file} ===")
        exe_path = compile_cuda_file(cuda_file)
        if exe_path is None:
            continue

        exe_name = os.path.basename(exe_path)
        print(f"Currently running: {exe_name}")

        # Special case for cuda_c_memory_test
        if cuda_file == "cuda_c_memory_test.cu":
            time_executable(exe_path, ["1"])
            time_executable(exe_path, ["2"])
        else:
            time_executable(exe_path)


if __name__ == "__main__":
    main()
