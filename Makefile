CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -pthread
LDFLAGS := -lportaudio -lv4l2 -lpthread -lm
TARGET := main_nocv_full
SRCS := main_nocv_full.cpp

all: $(TARGET)
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
