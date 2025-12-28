#!/bin/bash
g++ -shared -fPIC -o libov_engine.so src/ov_library.cpp -O3
echo "✅ Compiled libov_engine.so"
