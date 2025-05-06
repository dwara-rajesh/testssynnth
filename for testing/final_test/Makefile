# Makefile for ALSA Synth on BeagleBone

# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall \
	$(shell pkg-config --cflags opencv4 alsa)
LDFLAGS := $(shell pkg-config --libs opencv4 alsa) -pthread

# Target executable
TARGET := alsa_synth

# Source files
SRCS := AlsaSynth.cpp
OBJS := $(SRCS:.cpp=.o)

# Default rule
all: $(TARGET)

# Link
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
