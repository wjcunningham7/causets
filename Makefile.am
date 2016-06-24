AUTOMAKE_OPTIONS = foreign
ACLOCAL_AMFLAGS = -I m4 --install

SUBDIRS = printcolor inc src

INCD = -I $(top_builddir)/inc -I $(top_builddir)/printcolor
LIBD = -L $(top_builddir)/printcolor -lprintcolor

if USE_GPU
LNK_STRING = $(LNK_STR)
BIN_STRING = $(BIN_STR) $(INCD) $(LIBD)
else
LNK_STRING = $(LNK_STR) $(INCD) $(LIBD)
BIN_STRING = $(BIN_STR)
endif

.PHONY: link all dirs cleanbin

all: dirs link

dirs:
	@mkdir -p $(top_builddir)/bin

link:
	@rm -f $(top_builddir)/src/linked.o
	$(LNK_STRING)
	$(BIN_STRING)

cleanbin:
	rm -rf $(top_builddir)/bin

install-exec-hook:
	@mkdir -p $(prefix)/bin
	cp $(top_builddir)/bin/$(BINARY_NAME)* $(prefix)/bin
	cp -r $(top_builddir)/etc $(prefix)
	@./$(top_builddir)/etc/mkdat $(prefix)
	cp test README $(prefix)

uninstall-hook:
	rm $(prefix)/bin/$(BINARY_NAME)*
