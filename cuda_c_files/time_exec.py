import subprocess
import time
import statistics
import sys 

def time_executable(executable_path, args=None, runs=5):
    times = []
    for i in range(runs):
        print(f"Run {i + 1}/{runs}...")
        start = time.perf_counter()
        try:
            subprocess.run([executable_path] + (args or []), check=True)
        except subprocess.CalledProcessError as e:
            print(f"Execution failed on run {i + 1}: {e}")
            continue
        end = time.perf_counter()
        elapsed = end - start
        times.append(elapsed)
        print(f"Time: {elapsed:.4f} seconds")

    if times:
        mean_time = statistics.mean(times)
        std_dev = statistics.stdev(times)
        
        print(f"\nAverage execution time over {len(times)} successful runs: {mean_time:.4f} seconds")
        print(f"\nStandard deviation over {len(times)} successful runs: {std_dev:.4f} seconds")
    else:
        print("No successful runs.")


exec_name = sys.argv[1]
print(f"Currently running: {exec_name}")
time_executable(f"C:/Users/night/Documents/Praca magisterska/Merytoryczne/Idea/{exec_name}", runs=5) #Cold run
time_executable(f"C:/Users/night/Documents/Praca magisterska/Merytoryczne/Idea/{exec_name}", runs=10) #Actual test
