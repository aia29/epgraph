CXX := g++
INCLUDE_DIR := ../include/

default: example_0 example_1 example_2 example_3

example_0: example_0.cpp
	${CXX} example_0.cpp -o run_example_0 -I ${INCLUDE_DIR}
	
example_1: example_1.cpp
	${CXX} example_1.cpp -o run_example_1 -I ${INCLUDE_DIR}

example_2: example_2.cpp
	${CXX} example_2.cpp -o run_example_2 -I ${INCLUDE_DIR}

example_3: example_3.cpp
	${CXX} example_3.cpp -o run_example_3 -I ${INCLUDE_DIR}

clean:
	-rm -f run_example_*
