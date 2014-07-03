#!/usr/bin/perl
#
# compute the undirected and in- and out-degree distributions for a directed graph
#
# input:
#   graph links, one per line, in the format (node1 node2), from node1 to node2
#   in the file specified as the first command line argument;
#   the file can be gzipped (.gz)
# output:
#   degree distributions in the format (k N(k)),
#   in the files named ...total.stat.dat, ...in.stat.dat, and ...out.stat.dat.
#   the graph size, maximum and average degrees go to ...stat.sum.dat:
#   first line for the undirected, and then for in- and out-stats.

use strict;
use warnings;

# I/O stuff
my $infile = "";
my $outfile = "";
my $inoutfile = "";
my $outoutfile = "";
my $sumfile = "";
my $basedir = "../";
if (@ARGV) {
  $infile = $ARGV[0];
  $outfile = $basedir."dat/dst/".$infile;
  $infile =~ s/(.*\.gz)\s*$/gzip -dc < $1|/;
  $outfile =~ s/\.gz$//;
  $outfile =~ s/dat$//;
  $outfile =~ s/txt$//;
  $outfile =~ s/net$//;
  $outfile .= "." if $outfile ne "" and not $outfile =~ /\.$/; 
  $sumfile = $basedir."dat/".$outfile."stat.sum.dat";
  $inoutfile = $basedir."dat/idd/".$outfile."in.stat.dat";
  $outoutfile = $basedir."dat/odd/".$outfile."out.stat.dat";
  $outfile .= "total.stat.dat";
} else {
  die("no input file specified");
}

# compute the degrees of each node
my %nodeDegree = ();
open(INPUT, $infile) or die("can't open $infile: $!");
while(<INPUT>) {
  next if /^\s*$/ or /^\s*#/;
  chomp;
  die("input line \"$_\" is not a graph link") if not /^\s*(\S+)\s+(\S+)/;
  $nodeDegree{$1}[1]++; #out-degree
  $nodeDegree{$2}[0]++; #in-degree
}
close(INPUT);

# compute the degree distribution stats
my %deg = ();
my %indeg = ();
my %outdeg = ();
for my $node (keys %nodeDegree) {
    my $degree = 0;
    if (my $indegree = $nodeDegree{$node}[0]) {
        $indeg{$indegree}++;
        $degree += $indegree;
    }
    if (my $outdegree = $nodeDegree{$node}[1]) {
        $outdeg{$outdegree}++;
        $degree += $outdegree;
    }
    die("node $node has zero degree") if not $degree;
    $deg{$degree}++;
}

# print the statistics
my $n = keys %nodeDegree;
if ($n > 0) {
    open(SUMMARY, ">$sumfile") or die("can't open $sumfile: $!");
    # total degree
    my @degrees = sort {$a <=> $b} keys %deg;
    my $maxdeg = $degrees[$#degrees];
    my $avedeg = 0;
    open(OUTPUT, ">$outfile") or die("can't open $outfile: $!");
    for my $degree (@degrees) {
        my $nnodes = $deg{$degree};
        $avedeg += $degree*$nnodes;
        print OUTPUT "$degree $nnodes\n";
    }
    close(OUTPUT);
    $avedeg /= $n;
    print SUMMARY "$n $maxdeg $avedeg\n";
    # indegree
    @degrees = sort {$a <=> $b} keys %indeg;
    $maxdeg = $degrees[$#degrees];
    $avedeg = 0;
    $n = 0;
    open(OUTPUT, ">$inoutfile") or die("can't open $inoutfile: $!");
    for my $degree (@degrees) {
        my $nnodes = $indeg{$degree};
        $avedeg += $degree*$nnodes;
        $n += $nnodes;
        print OUTPUT "$degree $nnodes\n";
    }
    close(OUTPUT);
    $avedeg /= $n;
    print SUMMARY "$n $maxdeg $avedeg\n";
    # outdegree
    @degrees = sort {$a <=> $b} keys %outdeg;
    $maxdeg = $degrees[$#degrees];
    $avedeg = 0;
    $n = 0;
    open(OUTPUT, ">$outoutfile") or die("can't open $outoutfile: $!");
    for my $degree (@degrees) {
        my $nnodes = $outdeg{$degree};
        $avedeg += $degree*$nnodes;
        $n += $nnodes;
        print OUTPUT "$degree $nnodes\n";
    }
    close(OUTPUT);
    $avedeg /= $n;
    print SUMMARY "$n $maxdeg $avedeg\n";
    close(SUMMARY);
}
