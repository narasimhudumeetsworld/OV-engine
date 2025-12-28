CC = g++
CFLAGS = -I./kernels -std=c++17 -O3

all: ov_engine_full

ov_engine_full: src/ov_engine_core.cpp
	$(CC) $(CFLAGS) -o ov_engine_full src/ov_engine_core.cpp

clean:
	rm -f ov_engine_full
