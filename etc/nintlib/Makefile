ifneq (, $(findstring compute, $(HOSTNAME)))
LOCAL_DIR := /home/$(USER)/local
else ifneq (, $(findstring tiberius, $(HOSTNAME)))
LOCAL_DIR := /usr/local
else
$(error Hostname not recognized!)
endif

all : object lib

object : nintlib.cpp
	g++ -O3 -g -c -o nintlib.o nintlib.cpp

lib : 
	ar rvs libnint.a nintlib.o

install :
	mkdir -p $(LOCAL_DIR)/lib64
	mkdir -p $(LOCAL_DIR)/include/nintlib
	cp libnint.a $(LOCAL_DIR)/lib64/
	cp nintlib.hpp $(LOCAL_DIR)/include/nintlib/

clean :
	rm -f nintlib.o libnint.a
