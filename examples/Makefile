# Compiler(s)
CXX := g++

# Paths
SRC_DIRS := ./
BUILD_DIR := ./build/
EIGEN_DIR := ./eigen-3.4.0
INCLUDE_DIRS := ../include/ $(EIGEN_DIR)

# Build flags
INCLUDE_FLAGS := $(addprefix -I,$(INCLUDE_DIRS))
CXXFLAGS := -O2 -Wall -std=c++20 $(INCLUDE_FLAGS)

# Prepare lists
CXX_FILES := $(notdir $(shell find $(SRC_DIRS) -name "example_*.cpp"))
EXE_FILES := $(patsubst %.cpp, %, $(CXX_FILES))

all: $(EXE_FILES)
	$(shell mkdir -p $(BUILD_DIR))
	$(shell mv $(EXE_FILES) $(BUILD_DIR))

clean:
	rm -f $(EXE_FILES)
	rm -rf $(BUILD_DIR)
