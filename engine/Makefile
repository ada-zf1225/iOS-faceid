# Build the pure-C++ engine tests and CLI against the SAME source the iOS app ships.
CXX      ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pthread
ENGINE    = ../FaceID/FaceEngine.cpp
INC       = -I../FaceID

.PHONY: test cli clean
test:
	$(CXX) $(CXXFLAGS) $(INC) tests/test_face_engine.cpp $(ENGINE) -o /tmp/face_engine_tests
	/tmp/face_engine_tests

cli:
	$(CXX) $(CXXFLAGS) $(INC) cli/face_cli.cpp $(ENGINE) -o face_cli

clean:
	rm -f face_cli /tmp/face_engine_tests
