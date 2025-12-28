import ctypes
import numpy as np
import time
import threading
import os

# Load C++ Library
lib_path = os.path.abspath("libov_engine.so")
try:
    ov_lib = ctypes.CDLL(lib_path)
    
    # Define Signatures
    ov_lib.ov_cpu_graph_walk.argtypes = [
        ctypes.POINTER(ctypes.c_float), 
        ctypes.c_int, 
        ctypes.POINTER(ctypes.c_float)
    ]
    ov_lib.ov_cpu_graph_walk.restype = ctypes.c_float
    
    print("✅ Loaded C++ Engine")
except OSError:
    print("⚠️  Library not compiled. Run ./compile_lib.sh first.")
    ov_lib = None

def run_parallel_inference():
    print("\n🚀 Starting Hybrid CPU/GPU Inference Loop...")
    
    # Mock Data
    vector_dim = 4096
    query = np.random.rand(vector_dim).astype(np.float32)
    truth = np.random.rand(vector_dim).astype(np.float32)
    
    # Shared State
    results = {"confidence": 0.0}
    
    # 1. CPU TASK (Graph Walk)
    def cpu_worker():
        if ov_lib:
            start = time.time()
            # Convert numpy to C pointer
            q_ptr = query.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            t_ptr = truth.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            # Call C++
            conf = ov_lib.ov_cpu_graph_walk(q_ptr, vector_dim, t_ptr)
            results["confidence"] = conf
            print(f"   [CPU Thread] Graph Walk Finished: {conf:.4f} (took {time.time()-start:.6f}s)")
        else:
            time.sleep(0.01) # Mock work
            results["confidence"] = 0.85

    # 2. GPU TASK (Main Thread)
    # Start CPU immediately
    cpu_thread = threading.Thread(target=cpu_worker)
    cpu_thread.start()
    
    print("   [GPU Thread] Computing Attention Layers 0-10...")
    time.sleep(0.02) # Simulate GPU work
    
    # 3. SYNCHRONIZE
    cpu_thread.join()
    print(f"   [SYNC] CPU Data Ready. Injecting into Layer 11...")
    
    # 4. Correction
    print(f"   ✅ State Corrected with Confidence: {results['confidence']}")

if __name__ == "__main__":
    run_parallel_inference()
