CXX ?= g++
GXX ?= g++

CXXFLAGS += -std=c++11
INCFLAGS += -I/usr/local/Cellar/opencv\@2/2.4.13.6_2/include
LDFLAGS += -L/usr/local/lib 
LDFLAGS += -L/usr/local/Cellar/opencv\@2/2.4.13.6_2/lib 
LIBS += -lglog -lgflags -lprotobuf -lleveldb -lsnappy \
		   -llmdb -lboost_system -lm -lopencv_core -lopencv_highgui \
		   -lopencv_imgproc -lboost_thread

all: test_io

data.pb.h data.pb.cc: data.proto
	protoc data.proto --cpp_out=./

io.o: io.cpp io.hpp data.pb.h
	$(CXX) $(CXXFLAGS) -c -Wall io.cpp $(INCFLAGS)

db.o: db.cpp db.hpp data.pb.h
	$(CXX) $(CXXFLAGS) -c -Wall db.cpp $(INCFLAGS)

test_io: io.o db.o test_io.cpp data.pb.h data.pb.cc
	$(CXX) $(CXXFLAGS) data.pb.cc test_io.cpp io.o db.o -o test_io.out $(LDFLAGS) $(LIBS) $(INCFLAGS)

clean:
	rm -f io.o db.o test_io.out data.pb.h data.pb.cc
