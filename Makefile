CXX := clang
TORCH := $(HOME)/libtorch
ABI := 0
OS := $(shell uname)
CPPFLAGS := -I $(TORCH)/include -I $(TORCH)/include/torch/csrc/api/include 
CXXFLAGS := -std=c++17 -std=gnu++17 -pedantic -Wall -Wfatal-errors -fPIC -O3
LDFLAGS := -shared -L$(TORCH)/lib

# workaround for bug w'gcc (https://github.com/pytorch/pytorch/issues/60341)
ifeq ($(CXX),clang)
 LDLIBS := -l torch -Wl,-rpath $(TORCH)/lib
else ifeq ("$(wildcard $(TORCH)/lib/libtorch_cuda.so)", "")
 LDLIBS := -l torch -l torch_cpu -Wl,-rpath $(TORCH)/lib
else
 LDLIBS := -l torch -l torch_cpu -l torch_cuda -Wl,-rpath $(TORCH)/lib
endif

ifeq ($(OS),Linux)
 CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=$(ABI)
endif
ifeq ($(OS),Darwin)
 LDFLAGS := -undefined dynamic_lookup $(LDFLAGS)
endif

lib := ktorch.so
src := ktorch.cpp ktensor.cpp kmath.cpp knn.cpp kloss.cpp kopt.cpp kmodel.cpp ktest.cpp $(wildcard knn/*.cpp) $(wildcard kopt/*.cpp)

all: $(lib)
*.o: k.h ktorch.h private.h
*/*.o: k.h ktorch.h
ktorch.o: stb_image_write.h
kloss.o: kloss.h knn/distance.h

kopt.o: kopt.h $(wildcard kopt/*.h)
kopt/lamb.o: kopt/lamb.h

knn.o: knn.h $(wildcard knn/*.h)
knn/act.o: knn/act.h knn/fns.h knn/util.h
knn/attention.o: knn/attention.h knn/util.h
knn/callback.o: knn/callback.h knn/util.h
knn/conv.o: knn/conv.h knn/util.h
knn/distance.o: knn/distance.h knn/util.h
knn/drop.o: knn/drop.h knn/util.h
knn/embed.o: knn/embed.h knn/util.h
knn/fns.o: knn/fns.h knn/util.h
knn/fold.o: knn/fold.h knn/util.h
knn/fork.o: knn/fork.h knn/util.h
knn/linear.o: knn/linear.h knn/util.h
knn/nbeats.o: knn/nbeats.h knn/util.h
knn/norm.o: knn/norm.h knn/util.h
knn/onehot.o: knn/onehot.h knn/util.h
knn/pad.o: knn/pad.h knn/util.h
knn/recur.o: knn/recur.h knn/util.h
knn/reshape.o: knn/reshape.h knn/util.h
knn/residual.o: knn/residual.h knn/util.h
knn/select.o: knn/select.h knn/util.h
knn/seq.o: knn/seq.h knn/util.h
knn/squeeze.o: knn/squeeze.h knn/util.h
knn/transformer.o: knn/transformer.h knn/norm.h knn/util.h
knn/transform.o: knn/transform.h knn/util.h
knn/upsample.o: knn/upsample.h knn/util.h
knn/util.o: knn/util.h knn/util.h

$(lib): $(src:.cpp=.o)
	$(CXX) -o $@ $^ $(LDFLAGS) $(LDLIBS)

clean:
	$(RM) *.o $(wildcard knn/*.o) $(wildcard kopt/*.o) $(lib)
