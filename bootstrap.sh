#!/bin/sh

set -eu -o pipefail

install_fastmath=false
FASTMATH_BB="git@bitbucket.org:dk-lab/2015_code_fastmath.git"
if [[ ( "$#" -eq 1 && "$1" == "--reinstall-fastmath" ) ||  ! -d "opt/fastmath" ]] ; then
#  if [[ "$1" == "--reinstall-fastmath" ]] ; then
    rm -rf opt/fastmath
    install_fastmath=true
#  fi
fi

if [[ "${install_fastmath}" == "true" ]] ; then
  git clone $FASTMATH_BB opt/fastmath
fi

if [[ ! -d "opt/fastmath" ]] ; then
  echo "FastMath directory not found!"
  exit 1
fi

echo "bootstrap: Entering directory "'`'"$PWD/opt/fastmath'"
cd opt/fastmath
mkdir -p m4
touch NEWS AUTHORS ChangeLog
autoreconf -vfi
echo "bootstrap: Leaving directory "'`'"$PWD"
cd ../../

mkdir -p m4
touch NEWS AUTHORS ChangeLog
autoreconf -vfi
