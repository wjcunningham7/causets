AC_DEFUN([AX_PROG_MPICXX],
[
AC_PROVIDE([AX_PROG_MPICXX])

AC_ARG_ENABLE([mpi], AS_HELP_STRING([--enable-mpi], [Enable MPI capabilities (default no)]), [USE_MPI=$enable_mpi], [USE_MPI=no])

if [[[ $USE_MPI == "yes" ]]] ; then
  AC_ARG_WITH([mpi], AS_HELP_STRING([--with-mpi=PREFIX], [Prefix of your MPI installation]),
    [mpi_prefix=$with_mpi],
    [ if [[[ "$#" -ge 1 && "$1" != "" ]]] ; then
        mpi_prefix="$1"
      else
        mpi_prefix=$(which mpiCC | sed "s/\/bin\/mpiCC//")
      fi
    ])

  if [[[ $mpi_prefix != "" ]]] ; then
    AC_MSG_CHECKING([mpiCC in $mpi_prefix/bin])
    if [[[ -e $mpi_prefix/bin/mpiCC ]]] ; then
      AC_MSG_RESULT([found])
      MPI_HOME=$mpi_prefix
      MPICC=$mpi_prefix/bin/mpiCC
      VALID_MPI=yes
    else
      AC_MSG_RESULT([not found])
      AC_MSG_WARN([mpiCC was not found in $mpi_prefix/bin])
      VALID_MPI=no
    fi
  else
    AC_MSG_WARN([Could not find installation directory for MPI.])
    VALID_MPI=no
  fi
fi

if [[[ $USE_MPI == "yes" && $VALID_MPI == "yes" ]]] ; then
  AC_MSG_CHECKING([MPI version])
  ax_mpi_version=$($MPICC -dumpversion)
  AC_MSG_RESULT([$ax_mpi_version])
  AX_COMPARE_VERSION([$ax_mpi_version], [ge], [4.8.1],, AC_MSG_WARN([MPI version must be at least 4.8.1]); VALID_MPI=no)
fi


if [[[ $USE_MPI == "yes" && $VALID_MPI == "yes" ]]] ; then
  AC_CHECK_HEADER([mpi.h],, AC_MSG_WARN([Could not find mpi.h]); VALID_MPI=no)
fi

if [[[ $USE_MPI == "yes" ]]] ; then
  AC_MSG_CHECKING([if MPI will be used])
  if [[[ $VALID_MPI == "yes" ]]] ; then
    AC_MSG_RESULT([yes])
    AC_SUBST([MPI_ENABLED], [yes])

    if [[[ $(wc -w <<< "$CXX") -gt 1 ]]] ; then
      CXX="$MPICC "$(cut -d' ' -f2- <<< $CXX)
    else
      CXX=$MPICC
    fi
    MPI_FLAGS="-DMPI_ENABLED -Wno-deprecated"
    AC_SUBST([MPI_FLAGS], [$MPI_FLAGS])
  else
    AC_MSG_RESULT([no])
    AC_SUBST([MPI_ENABLED], [no])
    AC_MSG_ERROR([Could not properly enable MPI.])
  fi
fi
])
