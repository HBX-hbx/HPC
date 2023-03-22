CXXFLAGS := -O3 -std=c++11

.PHONY: clean

all: allreduce

allreduce: allreduce.cpp
	mpicxx $^ ${CXXFLAGS} -o $@

clean:
	rm allreduce
