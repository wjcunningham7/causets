AUTOMAKE_OPTIONS = foreign
ACLOCAL_AMFLAGS = -I m4 --install

SUBDIRS = printcolor inc src

INCD = -I $(top_builddir)/inc -I $(top_builddir)/printcolor
LIBD = -L $(top_builddir)/printcolor -lprintcolor
DATDIR = $(top_builddir)/dat
ETCDIR = $(top_builddir)/etc

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

cleandata :
	@ rm -f $(DATDIR)/*.cset.out $(DATDIR)/pos/*.cset.pos.dat $(DATDIR)/edg/*.cset.edg.dat $(DATDIR)/dst/*.cset.dst.dat $(DATDIR)/idd/*.cset.idd.dat $(DATDIR)/odd/*.cset.odd.dat $(DATDIR)/cls/*.cset.cls.dat $(DATDIR)/cdk/*.cset.cdk.dat $(DATDIR)/emb/*.cset.emb.dat $(DATDIR)/emb/tn/*.cset.emb_fn.dat $(DATDIR)/emb/fp/*.cset.emb_fp.dat $(ETCDIR)/data_keys.cset.key $(DATDIR)/ref/*.ref $(DATDIR)/idf/*.cset.idf.dat $(DATDIR)/odf/*.cset.odf.dat $(DATDIR)/act/*.cset.act.dat $(DATDIR)/vpd/*.cset.vpd.dat

install-exec-hook:
	@mkdir -p $(prefix)/bin
	cp $(top_builddir)/bin/$(BINARY_NAME)* $(prefix)/bin
	cp -r $(top_builddir)/etc $(prefix)
	@./$(top_builddir)/etc/mkdat $(prefix)
	cp test README $(prefix)

uninstall-hook:
	rm $(prefix)/bin/$(BINARY_NAME)*
