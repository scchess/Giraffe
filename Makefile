HAS_TORCH=1
TORCH_DIR=~/torch/install

WIN32_CC=i686-w64-mingw32-gcc
WIN64_CC=x86_64-w64-mingw32-gcc
WIN32_CXX=i686-w64-mingw32-g++
WIN64_CXX=x86_64-w64-mingw32-g++
WIN32_STRIP=i686-w64-mingw32-strip
WIN64_STRIP=x86_64-w64-mingw32-strip

ifndef CXX
	CXX=g++
endif

ifndef CC
	# this is used to build gtb only
	CC=gcc
endif

HGVERSION:= $(shell hg parents --template '{node|short}')

CXXFLAGS_BASE = \
	-Wall -Wextra -Wno-unused-function -std=gnu++11 -mtune=native -ffast-math \
	-pthread -fopenmp -DHGVERSION="\"${HGVERSION}\""

# we will then extend this one with optimization flags
CXXFLAGS:= $(CXXFLAGS_BASE)

CXXFLAGS_DEP = \
	-std=gnu++11

LDFLAGS=-L. -Lgtb -lm -lgtb

ifeq ($(HAS_TORCH), 1)
	LDFLAGS += -L $(TORCH_DIR)/lib
	
	# for LuaJIT
	LDFLAGS += -lluaT -lTH -lluajit
	
	#for LUA 5.2
	#LDFLAGS += -lluaT -lTH -llua
	
	CXXFLAGS += -DHAS_TORCH -I $(TORCH_DIR)/include
endif

ifeq ($(PG), 1)
	CXXFLAGS += -g -Os -pg
else ifeq ($(DEBUG),1)
	CXXFLAGS += -g -Og
else
	CXXFLAGS += -O3 -flto -march=native
endif

CXXFILES := \
	$(wildcard *.cpp) \
	$(wildcard ann/*.cpp) \
	$(wildcard eval/*.cpp) \
	$(wildcard move_stats/*.cpp)

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
	$(Q) cd gtb && CC=$(CC) CFLAGS=$(CFLAGS) make

test:
	$(Q) echo $(DEPS)
	
clean:
	-$(Q) rm -f $(DEPS) $(OBJS) $(EXE)
	$(Q) cd gtb && make clean
	
windows:
	$(Q) cd gtb && make clean && make CFLAGS=-m32 CC=$(WIN32_CC)
	$(WIN32_CXX) $(CXXFLAGS_BASE) $(INCLUDES) -O3 -static -march=pentium2 $(CXXFILES) -o giraffe_w32.exe -Lgtb -lgtb -pthread -static-libgcc -static-libstdc++
	$(WIN32_STRIP) -g -s giraffe_w32.exe
	$(Q) cd gtb && make clean && make CFLAGS=-m64 CC=$(WIN64_CC)
	$(WIN64_CXX) $(CXXFLAGS_BASE) $(INCLUDES) -O3 -static -march=nocona $(CXXFILES) -o giraffe_w64.exe -Lgtb -lgtb -pthread -static-libgcc -static-libstdc++
	$(WIN64_STRIP) -g -s giraffe_w64.exe

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

# -march benchmark notes:
# SSE = +37%
# SSE2 = +10%
# little improvement from SSE3
