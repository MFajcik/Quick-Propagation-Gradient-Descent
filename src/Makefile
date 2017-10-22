#init
#### info
#
# 'make'       
# 'make clean'  removes all .o and executable files
# @2015 Martin Fajcik FIT_VUT v Brne    xfajci00@stud.fit.vutbr.cz 
CC=g++
CFLAGS = -Wall  -pedantic  -Wextra -W -g -O3 `pkg-config --cflags sdl cairo` -std=c++11 -pthread
CLIBS+=-L/usr/lib/x86_64-linux-gnu -lm `pkg-config --libs sdl cairo` -lpthread
OBJ = quickprop.o
HEADERS = neuron.h

all: quickprop.o quickprop

quickprop.o : QuickProp.cpp $(HEADERS)
	$(CC)  -c -o $@ $< $(CFLAGS) $(CLIBS)

quickprop   : $(OBJ)
	$(CC)  -o $@ $(OBJ) $(CFLAGS) $(CLIBS)
			
clean :
	@rm -f *.o *~ quickprop


