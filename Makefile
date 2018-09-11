CXX ?= g++-5
GXX ?= g++-5

HALIDE_PATH = /Users/tianjunz/Desktop/llvm5.0/halide_build
CXXFLAGS += -std=c++11 -g

INCFLAGS += -I$(HALIDE_PATH)/include
INCFLAGS += -I/usr/local/Cellar/opencv\@2/2.4.13.6_2/include
INCFLAGS += -I/Users/tianjunz/Desktop/llvm5.0/Halide/tools
LDFLAGS += -L/usr/local/lib -L/usr/local/Cellar/opencv\@2/2.4.13.6_2/lib 
LIBS += -lglog -lgflags -lprotobuf -lleveldb -lsnappy \
	-llmdb -lboost_system -lm -lopencv_core -lopencv_highgui \
	-lopencv_imgproc -lboost_thread -ldl -lpthread -lz \
	-lHalide -L$(HALIDE_PATH)/bin

all: test conv_bench vgg_bench vgg_bench_train

utils.o: utils.h utils.cpp
	$(CXX) $(CXXFLAGS) utils.cpp -c -Wall $(INCFLAGS)

test: layer_test.cpp layers.h utils.o dataloaders/io.o dataloaders/db.o
	$(CXX) $(CXXFLAGS) layer_test.cpp dataloaders/data.pb.cc dataloaders/io.o dataloaders/db.o utils.o -o layer_test.out $(LDFLAGS) $(LIBS) $(INCFLAGS)

conv_bench: conv_bench.cpp layers.h utils.o dataloaders/io.o dataloaders/db.o
	$(CXX) $(CXXFLAGS) conv_bench.cpp dataloaders/data.pb.cc dataloaders/io.o dataloaders/db.o utils.o -o conv_bench.out $(LDFLAGS) $(LIBS) $(INCFLAGS)

vgg_bench: vgg_forward.cpp layers.h utils.o dataloaders/io.o dataloaders/db.o
	$(CXX) $(CXXFLAGS) vgg_forward.cpp dataloaders/data.pb.cc dataloaders/io.o dataloaders/db.o utils.o -o vgg.out $(LDFLAGS) $(LIBS) $(INCFLAGS)

vgg_bench_train: vgg_forward_train.cpp layers.h utils.o dataloaders/io.o dataloaders/db.o
	$(CXX) $(CXXFLAGS) vgg_forward_train.cpp dataloaders/data.pb.cc dataloaders/io.o dataloaders/db.o utils.o -o vgg_train.out $(LDFLAGS) $(LIBS) $(INCFLAGS)

clean:
	rm -f layer_test.out utils.o conv_bench.out vgg.out vgg_train.out
