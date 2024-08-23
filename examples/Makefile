# Compiler(s)
CXX := g++

# Paths
SRC_DIRS := ./
BUILD_DIR := ./build/
EIGEN_DIR := ./eigen-3.4.0
INCLUDE_DIRS := ../include/ $(EIGEN_DIR)

# Build flags
INCLUDE_FLAGS := $(addprefix -I,$(INCLUDE_DIRS))
CXXFLAGS := -O2 -std=c++20 $(INCLUDE_FLAGS)

# Prepare lists
CXX_FILES := $(notdir $(shell find $(SRC_DIRS) -name "example_*.cpp"))
EXE_FILES := $(patsubst %.cpp, $(BUILD_DIR)%, $(CXX_FILES))


all: EIGEN BUILD_DIR $(EXE_FILES)

EIGEN:
ifeq (,$(wildcard ./eigen-3.4.0))
ifeq (,$(wildcard ./eigen-3.4.0.tar.gz))
	$(shell wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz)
endif
	$(shell tar -xf eigen-3.4.0.tar.gz)
endif

BUILD_DIR:
	$(shell mkdir -p $(BUILD_DIR))

$(EXE_FILES):
	$(CXX) $(CXXFLAGS) $(notdir $@).cpp -o $@

clean:
	rm -f $(EXE_FILES)
	rm -rf $(BUILD_DIR)
	rm -f van_der_pol.csv
