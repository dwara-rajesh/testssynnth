CXX      := g++
CXXFLAGS := -std=c++17 -O3 `pkg-config --cflags opencv4 portaudio-2.0` -pthread
LDFLAGS  := `pkg-config --libs opencv4 portaudio-2.0` -lv4l2

TARGET   := bb_opencv_synth
SRCS     := main.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
