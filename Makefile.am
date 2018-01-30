###########################
#(C) Will Cunningham 2017 #
#         DK Lab          #
# Northeastern University #
###########################

AUTOMAKE_OPTIONS = foreign
ACLOCAL_AMFLAGS = -I m4 --install

SUBDIRS = inc src

INCD = -I $(top_builddir)/inc -I $(top_builddir)/src/printcolor
LIBD = -L $(top_builddir)/src/printcolor -lprintcolor
DATDIR = $(top_builddir)/dat
ETCDIR = $(top_builddir)/etc

if USE_GPU
LNK_STRING = $(LNK_STR)
BIN_STRING = $(BIN_STR) $(INCD) $(LIBD)
else
LNK_STRING = $(LNK_STR) $(INCD) $(LIBD)
BIN_STRING = $(BIN_STR)
endif

.PHONY: all dirs link cleanbin

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
	@ rm -f $(DATDIR)/*.cset.out $(DATDIR)/pos/*.cset.pos.dat $(DATDIR)/edg/*.cset.edg.dat $(DATDIR)/dst/*.cset.dst.dat $(DATDIR)/idd/*.cset.idd.dat $(DATDIR)/odd/*.cset.odd.dat $(DATDIR)/cls/*.cset.cls.dat $(DATDIR)/cdk/*.cset.cdk.dat $(DATDIR)/emb/*.cset.emb.dat $(DATDIR)/emb/tn/*.cset.emb_fn.dat $(DATDIR)/emb/fp/*.cset.emb_fp.dat $(ETCDIR)/data_keys.cset.key $(DATDIR)/ref/*.ref $(DATDIR)/idf/*.cset.idf.dat $(DATDIR)/odf/*.cset.odf.dat $(DATDIR)/act/*.cset.act.dat $(DATDIR)/vpd/*.cset.vpd.dat $(DATDIR)/dot/*.cset.dot $(DATDIR)/fol/as/*.cset.fol.dat $(DATDIR)/fol/tl/*.cset.fol.dat $(DATDIR)/fol/sl/*.cset.fol.dat

distclean-local:
	rm -rf $(top_builddir)/{autom4te.cache,aclocal.m4,AUTHORS,ChangeLog,config.guess,config.log,config.status,config.sub,configure,depcomp,install-sh,Makefile,Makefile.in,missing,NEWS,src/Makefile.in,src/printcolor/Makefile.in,inc/Makefile.in,inc/config.h*,inc/stamp-h1,bin}

install-exec-hook:
	@mkdir -p $(prefix)/bin
	cp $(top_builddir)/bin/$(BINARY_NAME)* $(prefix)/bin
	cp -r $(top_builddir)/etc $(prefix)
	@./$(top_builddir)/etc/mkdat $(prefix)

uninstall-hook:
	rm $(prefix)/bin/$(BINARY_NAME)*
