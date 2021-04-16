CXX := clang++
TORCH := $(HOME)/libtorch
ABI := 0
OS := $(shell uname)
CPPFLAGS := -isystem $(TORCH)/include -isystem $(TORCH)/include/torch/csrc/api/include 
CXXFLAGS := -std=c++14 -std=gnu++14 -pedantic -Wall -Wfatal-errors -fPIC -O3
LDFLAGS := -shared -L$(TORCH)/lib
LDLIBS := -l torch -Wl,-rpath $(TORCH)/lib

ifeq ($(OS),Linux)
 CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=$(ABI)
endif
ifeq ($(OS),Darwin)
 LDFLAGS := -undefined dynamic_lookup $(LDFLAGS)
endif

lib := ktorch.so
src := ktorch.cpp ktensor.cpp kmath.cpp knn.cpp kloss.cpp kopt.cpp kmodel.cpp ktest.cpp

all: $(lib)
*.o: k.h ktorch.h knn.h private.h
ktorch.o: stb_image_write.h
kloss.o: kloss.h

$(lib): $(subst .cpp,.o,$(src))
	$(CXX) -o $@ $^ $(LDFLAGS) $(LDLIBS)

clean:
	$(RM) *.o $(lib)
