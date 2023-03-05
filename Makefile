#!/bin/make -f 

CXX=g++
CC=gcc
CPPFLAGS=-g -Wall -pedantic-errors -std=c++2a -O2 -DBOOST_ALL_DYN_LINK
LDFLAGS=-lboost_program_options -lboost_log_setup -lboost_log -pthread -lboost_thread

SRCS=$(wildcard cpp/*.cc)
OBJS=$(subst .cc,.o,$(SRCS))

#$(info [$(SRCS)])

all: gen_solver

gen_solver: $(OBJS)
	$(CXX) $(CPPFLAGS) -o gen_solver $(OBJS) $(LDFLAGS)

clean:
	rm cpp/*.o 

