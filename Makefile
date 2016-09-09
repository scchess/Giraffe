CXX=g++-5

# this is used to build gtb only
CC=gcc-5

TORCH_DIR=~/torch/install

HGVERSION:= $(shell hg parents --template '{node|short}')

CXXFLAGS_BASE = \
	-Wall -Wextra -Wno-unused-function -std=gnu++11 -mtune=native -Wa,-q -ffast-math \
	-pthread -fopenmp -DHGVERSION="\"${HGVERSION}\"" -I $(TORCH_DIR)/include

# we will then extend this one with optimization flags
CXXFLAGS:= $(CXXFLAGS_BASE)
	
CXXFLAGS_DEP = \
	-std=gnu++11

LDFLAGS=-L. -L $(TORCH_DIR)/lib -Lgtb -lm -lgtb

# for LuaJIT
LDFLAGS += -lluaT -lTH -lluajit

#for LUA 5.2
#LDFLAGS += -lluaT -lTH -llua

ifeq ($(PG), 1)
	CXXFLAGS += -g -Og -pg
else ifeq ($(DEBUG),1)
	CXXFLAGS += -g -Og
else
	CXXFLAGS += -O3 -flto
endif

ifeq ($(CLUSTER), 1)
	CXXFLAGS += -march=sandybridge -static
	LDFLAGS += -Wl,--whole-archive -lpthread -Wl,--no-whole-archive
	LDFLAGS := $(filter-out -ltcmalloc,$(LDFLAGS))
else
	CXXFLAGS += -march=native
endif

CXXFILES := \
	$(wildcard *.cpp) \
	$(wildcard ann/*.cpp) \
	$(wildcard eval/*.cpp)

INCLUDES=-I.

EXE=giraffe

OBJS := $(CXXFILES:%.cpp=obj/%.o)
DEPS := $(CXXFILES:%.cpp=dep/%.d)

ifeq ($(V),0)
	Q = @
else
	Q =
endif

ifeq ($(OS),Windows_NT)
	# mingw builds crash with LTO
	CXXFLAGS := $(filter-out -flto,$(CXXFLAGS))
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
	endif
	ifeq ($(UNAME_S),Darwin)
		# OSX needs workaround for AVX, and LTO is broken
		# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=47785
		CXXFLAGS += -Wa,-q
		CXXFLAGS := $(filter-out -flto,$(CXXFLAGS))
		# luajit requires this on OS X
		LDFLAGS += -pagezero_size 10000 -image_base 100000000
	endif
endif

.PHONY: clean test windows

default: $(EXE)

dep/%.d: %.cpp
	$(Q) $(CXX) $(CXXFLAGS_DEP) $(INCLUDES) $< -MM -MT $(@:dep/%.d=obj/%.o) > $@
	
obj/%.o: %.cpp
	$(Q) $(CXX) $(CXXFLAGS) $(INCLUDES) -c $(@:obj/%.o=%.cpp) -o $@

$(EXE): $(OBJS) gtb/libgtb.a
	$(Q) $(CXX) $(CXXFLAGS) $(OBJS) -o $(EXE) $(LDFLAGS)

gtb/libgtb.a:
	$(Q) cd gtb && CC=$(CC) make

test:
	$(Q) echo $(DEPS)
	
clean:
	-$(Q) rm -f $(DEPS) $(OBJS) $(EXE)
	
windows:
	$(Q) cd gtb && make windows_clean && make CFLAGS=-m32
	g++ $(CXXFLAGS_BASE) -m32 $(INCLUDES) -O3 -static $(CXXFILES) -o giraffe_w32.exe -Lgtb -lgtb
	strip -g -s giraffe_w32.exe
	$(Q) cd gtb && make windows_clean && make CFLAGS=-m64
	g++ $(CXXFLAGS_BASE) -m64 $(INCLUDES) -O3 -static $(CXXFILES) -o giraffe_w64.exe -Lgtb -lgtb
	strip -g -s giraffe_w64.exe

no_deps = 
ifeq ($(MAKECMDGOALS),clean)
	no_deps = yes
endif

ifeq ($(MAKECMDGOALS),windows)
	no_deps = yes
endif
	
ifndef no_deps
	-include $(DEPS)
endif
