##########################
#(C) Will Cunningham 2017 #
#         DK Lab          #
# Northeastern University #
###########################

AUTOMAKE_OPTIONS = foreign
ACLOCAL_AMFLAGS = -I m4 --install

SUBDIRS = inc src

INCD = -I $(top_builddir)/inc
DATDIR = $(top_builddir)/data
ETCDIR = $(top_builddir)/etc

if USE_GPU
LNK_STRING = $(LNK_STR)
BIN_STRING = $(BIN_STR) $(INCD)
else
LNK_STRING = $(LNK_STR) $(INCD)
BIN_STRING = $(BIN_STR)
endif

.PHONY: all dirs link cleanbin
.NOTPARALLEL:

all: dirs link

dirs:
	mkdir -p $(top_builddir)/bin
	rm -f $(top_builddir)/bin/$(BINARY_NAME)

link:
	$(LNK_STRING)
	$(BIN_STRING)

cleanbin:
	rm -rf $(top_builddir)/bin

cleandata :
	@rm -rf $(DATDIR)

distclean-local:
	rm -rf $(top_builddir)/{autom4te.cache,aclocal.m4,AUTHORS,ChangeLog,config.guess,config.log,config.status,config.sub,configure,depcomp,install-sh,Makefile,Makefile.in,missing,NEWS,src/Makefile.in,inc/Makefile.in,inc/config.h*,inc/stamp-h1,bin,compile,dat,etc/compress,etc/compile,testbed,src/linked.obj}

install-exec-hook:
	@mkdir -p $(prefix)/bin
	cp $(top_builddir)/bin/* $(prefix)/bin
	cp -r $(top_builddir)/etc $(prefix)
	@./$(top_builddir)/etc/mkdat $(prefix)

uninstall-hook:
	rm $(prefix)/bin/*
