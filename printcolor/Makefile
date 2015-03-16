ifneq (, $(findstring compute, $(HOSTNAME)))
LOCAL_DIR := /home/$(USER)/local
else ifneq (, $(findstring tiberius, $(HOSTNAME)))
LOCAL_DIR := /usr/local
else
$(error Hostname not recognized!)
endif

all : object lib

object : printcolor.cpp
	g++ -O3 -g -c -o printcolor.o printcolor.cpp

lib : printcolor.o
	ar rvs libprintcolor.a printcolor.o

install :
	mkdir -p $(LOCAL_DIR)/lib64
	mkdir -p $(LOCAL_DIR)/include/printcolor
	cp libprintcolor.a $(LOCAL_DIR)/lib64/
	cp printcolor.h $(LOCAL_DIR)/include/printcolor/

clean : 
	rm -f printcolor.o libprintcolor.a
