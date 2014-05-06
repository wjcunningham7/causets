#!/bin/bash
echo "Initializing 'Causet' directory structure..."
mkdir src
mkdir inc
mkdir bin
mkdir dat
mkdir dat/cdk
mkdir dat/cls
mkdir dat/dst
mkdir dat/edg
mkdir dat/idd
mkdir dat/odd
mkdir dat/pos
echo "Completed"

echo "Retrieving remote git repository..."
git clone localhost:/var/repos/git/causets.git
git config --global core.editor vim
git config --global merge.tool vimdiff
echo "Completed"