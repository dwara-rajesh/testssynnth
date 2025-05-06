CXX:=g++
CXXFLAGS:=-std=c++17 -O2 -Wall -pthread
LDFLAGS:=-lportaudio -lv4l2 -lpthread -lm
TARGET:=main
SRCS:=$(TARGET:=.cpp)
all:$(TARGET)
$(TARGET):$(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
clean:
	rm -f $(TARGET)
