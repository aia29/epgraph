.PHONY : all

# Compiler(s)
CXX := g++

# Paths
BUILD_DIR := ./build/
SRC_DIRS := ./
INCLUDE_DIRS := ../include/

# Build flags
CXX_CUSTOM_FLAGS := -O2 -std=c++20
INCLUDE_FLAGS := $(addprefix -I,$(INCLUDE_DIRS))
CXX_FLAGS := $(CXX_CUSTOM_FLAGS) $(INCLUDE_FLAGS)

# Prepare lists
CXX_FILES := $(notdir $(shell find $(SRC_DIRS) -name "example_*.cpp"))
EXE_FILES := $(patsubst %.cpp, $(BUILD_DIR)%, $(CXX_FILES))

all: $(EXE_FILES)

$(EXE_FILES): $(CXX_FILES)
	$(shell mkdir -p $(BUILD_DIR))
	$(CXX) $(CXX_FLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)
