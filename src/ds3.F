c     generates causets sprinkled over 3+1-dimensional de sitter space
c     of size NODOS and average degree xk


      implicit double precision(x,r,d)
#ifdef MPI
      include 'mpif.h' !RSS
      character(len=12) filenameoutput
#endif
      character*80 buffer
      integer idum, NODOS
      integer process, nproc, myid, mcw, mpi_err   !RSS
      integer*8 lm, lnodos, niter, niterpp, count  !RSS
      integer, allocatable   :: istart(:), iend(:) !RSS
      integer*8, allocatable :: mycount(:)         !RSS
c     For larger graphs change these parameters
      parameter (NODOSMAX=10000000,NEDGESMAX=1500000)
c     The volume element of the 3-sphere is dV=Sin^2(xtheta1)*Sin(xtheta2)*dtheta1*dtheta2*dphi
C     Angular coordinates of nodes in a 3-sphere !RSS Note use of allocatable arrays
      real*8 xtheta1 ! 1st angular coor - 0 to Pi
      real*8 xtheta2 ! 2nd angular coor - 0 to Pi
      real*8 xphi    ! 3rd angular coor -  0 to 2Pi
      real*8, allocatable :: xeta(:)    ! conformal time of nodes
      real*8, allocatable :: xcrd(:,:)  ! coor of 3-D sphere in a 4D Euclidean space
      real*8 dkc_acos, dkc_tol
      real*8 dkc15_acos, dkc15_tol

      common xpi, xtau0

      external kon, xFtheta1, xFtau

      xetagen(x) = dacos(1./dcosh(dnewton(xFtau,x,xtau0,xprecision))) ! generator of eta's
      xtheta1gen(x) = dnewton(xFtheta1,x,xpi/2.,xprecision) ! generator of theta1's
      xtheta2gen(x) = dacos(1.-2.*x) ! generator of theta2's

      xpi        = 3.141592653589793
      xprecision = 1.e-14

      dkc_tol   = 0.0065 !RSS Maximum error in DKC approcimation to ACOS
      dkc15_tol = 0.1    !RSS Maximum error in DKC approcimation to ACOS

#ifdef MPI
      mcw = mpi_comm_world
      call mpi_init(mpi_err)
      call mpi_comm_size(mcw, nproc, mpi_err)
      call mpi_comm_rank(mcw, myid, mpi_err)
#else
      nproc = 1
      myid  = 0
#endif

c     Command Line arguments (only read by root process)
      if(myid == 0) then
         if(iargc() < 3) then
            write(0,*)
            write(0,*) 'Insufficient number of command line args'
            write(0,*) 'Usage: ds3 idum n k [a]'
            write(0,*) 'where idum is a random negative integer seed,'
            write(0,*) '  n the number of nodes'
            write(0,*) '  k the average degree,'
            write(0,*) '  a the pseudradius'
            write(0,*) '    [if a is not provided, then a = 1]'
#ifdef MPI
            call mpi_abort(mcw, mpi_err)
#endif
            stop
         endif
         if (iargc() >= 3) then
            call getarg(1,buffer)
            read(buffer,*) idum
            call getarg(2,buffer)
            read(buffer,*) NODOS
            call getarg(3,buffer)
            read(buffer,*) xk
         endif
         if(iargc() >= 4) then
            call getarg(4,buffer)
            read(buffer,*) xa
         else
            xa = 1.0
         endif
      endif

      allocate(xcrd(4,NODOS), xeta(NODOS))

c     calculate the current times, and sprinkling density
      x0     = 1.5 ! never change this in fear of diverging newton
      xeta0  = dnewton(kon,xk/NODOS,x0,xprecision)
      xtau0  = dacosh(1./dcos(xeta0))
      xt0    = xtau0*xa
      xdelta = 3.*NODOS/2./xpi**2/xa**4
     $     / dtan(xeta0)/(2.+1./dcos(xeta0)**2)

      if(myid == 0) then
         write(0,*) 'Output from de sitter simulations:'
         write(0,*) 'number of MPI processes = ', nproc
         write(0,*) 'random seed             = ', idum
         write(0,*) 'number of nodes         = ', NODOS
         write(0,*) 'average degree          = ', xk
         write(0,*) 'time eta0               = ', xeta0
         write(0,*) 'time tau0               = ', xtau0
         write(0,*) 'time t0                 = ', xt0
         write(0,*) 'sprinkling density      = ', xdelta
         write(0,*)
      endif

#ifdef MPI
c     open output file for each process
      write(filenameoutput,'(A,I5.5)') 'ds3causet_', myid
      open(21, file=filenameoutput)

      call mpi_bcast(idum, 1, mpi_integer,          0,
     $     mcw, mpi_err)
      call mpi_bcast(xn,   1, mpi_double_precision, 0,
     $     mcw, mpi_err)
      call mpi_bcast(xk,   1, mpi_double_precision, 0,
     $     mcw, mpi_err)
      call mpi_bcast(xa,   1, mpi_double_precision, 0,
     $     mcw, mpi_err)
#endif

C     Assign angular coors and conformal time to nodes
      do i=1,NODOS
         xphi    = 2.0*xpi*ran2(idum)
         xtheta1 = xtheta1gen(ran2(idum))
         xtheta2 = xtheta2gen(ran2(idum))
         xeta(i) = xetagen(ran2(idum))
         xcrd(1,i) = sin(xtheta1)*sin(xtheta2)*sin(xphi)
         xcrd(2,i) = sin(xtheta1)*sin(xtheta2)*cos(xphi)
         xcrd(3,i) = sin(xtheta1)*cos(xtheta2)
         xcrd(4,i) = cos(xtheta1)
      enddo

#ifdef MPI
      if(myid == 0) then
         write(0,*)
         write(0,*) '---- Done with assignments of coordinates ----'
         write(0,*)
         write(0,*) '---- Confirm that each process has the    ----'
         write(0,*) '---- same set of values for last node     ----'
         write(0,*)
      endif

      do i=0,nproc
         if(myid == i) then
            write(0,300) 'myid = ', myid, ' node: ', xcrd(1,nodos),
     +           xcrd(2,nodos), xcrd(3,nodos), xcrd(4,nodos)
 300        format(A, I5, A, f10.7, f10.7, f10.7, f10.7)
         endif
         call mpi_barrier(mcw, mpi_err)
      enddo
      call mpi_barrier(mcw, mpi_err)

C     --- Start of code by RSS to determine division of work ---
      lnodos = nodos            ! implicit cast to long integer
      lnproc = nproc            ! implicit cast to long integer

      niter   = (lnodos*(lnodos-1))/2
      niterpp = niter/lnproc
      if(mod(niterpp,lnproc) > 0) then
         niterpp = niterpp + 1
      endif

      if(myid == 0) then
         write(0,*) 'Number of graph nodes   = ', nodos
         write(0,*) 'Number of MPI processes = ', nproc
         write(0,*) '#iterations             = ', niter
         write(0,*) '#iterations/process     = ', niterpp
         write(0,*)
      endif

      allocate(istart(0:nproc-1))
      allocate(iend(0:nproc-1))
      allocate(mycount(0:nproc-1))

      count      = 0
      process    = 0
      istart(0)  = 0

      do i=0,nodos-2
         count = count + (nodos-1-i)
         if(count >= niterpp) then
            iend(process) = i
            mycount(process) = count
            count = 0
            process = process + 1
            istart(process) = i + 1
         endif
      enddo

      iend(nproc-1) = nodos-2
      mycount(nproc-1) = count

c     Original logic developed in C - converting to Fortran indexing
      do i=0,nproc-1
         istart(i)  = istart(i) + 1
         iend(i)    = iend(i)   + 1
      enddo

      if(myid == 0) then
         write(0,*) 'myid    first i    last i   # iterations'
         write(0,*) '-----   --------   -------- --------------'
         do i=0,nproc-1
            write(0,1000) i, istart(i), iend(i), mycount(i)
         enddo
 1000    format(I5, I10, I10, I16)
      endif
C     --- End of code by RSS to determine division of work ---

C     Note that each process only works on a subset of outer loop iterations
C     Loop bounds (1,NODOS-1) replaced by (istart(myid),iend(myid))

      do i=istart(myid),iend(myid)
#else
      do i=1,NODOS-1
#endif
         do j=i+1,NODOS
            xprod = xcrd(1,i)*xcrd(1,j) +
     $           xcrd(2,i)*xcrd(2,j) +
     $           xcrd(3,i)*xcrd(3,j) +
     $           xcrd(4,i)*xcrd(4,j)
            xangle_ij = dkc15_acos(xprod) ! fast ACOS approx
            xeta_i = xeta(i)
            xeta_j = xeta(j)
            if(xangle_ij < abs(xeta_i-xeta_j) + dkc15_tol) then
               xangle_ij = dacos(xprod) ! accurate ACOS
               if(xangle_ij > xpi) write(0,*) 'beware angle above pi!!'
               if(xangle_ij < abs(xeta_i-xeta_j)) then
#ifdef MPI
                  if(xeta_i > xeta_j) write(21,100) j,i
                  if(xeta_j > xeta_i) write(21,100) i,j
#else
                  if(xeta_i > xeta_j) write(*,100) j,i
                  if(xeta_j > xeta_i) write(*,100) i,j
#endif
               endif
            endif
         enddo
      enddo
#ifdef MPI
      close(21)
#endif
      stop
 100  format(I7,1x,I7) !RSS Change I6 --> I7 to allow for larger problems
      end




c     ---------- Functions start here ----------

      subroutine kon(x,fx,dfx)
c     compute the ratio between average degree and number of nodes (fx)
c     and its derivative (dfx) at conformal time eta = x
      double precision x, fx, dfx, xpi, dfxrt
      common xpi

      dfxrt = 1.1117829402861223 ! the root of dfx

      if(x < dfxrt) then
         write(0,*) 'too large ratio between <k> and N'
         stop
      endif

      fx = 2. * (-7. + 12.*x/dtan(x) + 12.*dlog(1./dcos(x))
     $     + (-5. + 6.*dlog(1./dcos(x)))/dcos(x)**2)
     $     / 3./xpi /(2.+1./dcos(x)**2)**2 /dtan(x)
      dfx = (59. + 34.*dcos(2.*x) + 3.*dcos(4.*x)
     $     - 96.*x*dcos(x)**2/dtan(x)
     $     + 24.*(2.+dcos(2.*x))*dlog(dcos(x)))
     $     / 2./xpi /(2.+dcos(2.*x))**3 /dtan(x)**2
      end subroutine kon

      subroutine xFtheta1(x,fx,dfx)
c     compute the CDF for theta1 (fx) and its derivative (dfx)
c     at theta1 = x
      double precision x, fx, dfx, xpi
      common xpi

      fx = (x - 0.5*dsin(2.*x))/xpi
      dfx = 2.*dsin(x)**2/xpi
      end subroutine xFtheta1

      subroutine xFtau(x,fx,dfx)
c     compute the CDF for tau: fx = Ftau(x)
c     and its derivative: dfx = rhotau(x)
      double precision x, fx, dfx, xpi, xtau0
      common xpi, xtau0

      fx = (9.*dsinh(x) + dsinh(3.*x))
     $   / (9.*dsinh(xtau0) + dsinh(3.*xtau0))
      dfx = 12.*dcosh(x)**3
     $    / (9.*dsinh(xtau0) + dsinh(3.*xtau0))

c     other options diverge here or there:
c     CDF for eta Feta(x):
c      fx = (2. + 1./dcos(x)**2)*dtan(x)
c     $   / (2. + 1./dcos(xeta0)**2)/dtan(xeta0)
c     or its log:
c      fx = dlog(fx)
c     rhoeta(x) = d(Feta(x))/dx:
c      dfx = 3. / dcos(x)**4
c     $   / (2. + 1./dcos(xeta0)**2)/dtan(xeta0)
c     rhoeta(x)/Feta(x) = d(log(Feta(x)))/dx:
c      dfx = 3. / dsin(x) / dcos(x)
c     $   / (2. + dcos(2.*x))
c     d(1/Feta(x))/dx:
c      dfx = - 3. * (2. + dcos(2.*xeta0))*dtan(xeta0)
c     $    / dcos(xeta0)**2/dtan(x)**2/(2.+dcos(2.*x))**2
c     rhotau(x)/Ftau(x) = d(log(Ftau(x)))/dx:
c      dfx = 6.*dcosh(x)**2/dtanh(x)/(5.+dcosh(2.*x))
      end subroutine xFtau


      function dnewton(f,z,x,p)
c     newton method to solve f(x) = z
c     x - initial guess, p - precision
      double precision x, z, p, dnewton
      double precision fx, dfx, xn, xo
      integer i, maxi
      maxi = 200

      xo = x
      do i = 1,maxi
         call f(xo,fx,dfx) ! fx = f(x); dfx = f'(x)
         if(abs(dfx) < p) then
            write(0,*) 'derivative is too small'
            stop
         endif
         xn = xo + (z-fx)/dfx
         if(abs(xn-xo) < p) then
            dnewton = xn
            return
         endif
         xo = xn
      enddo
      write(0,*) 'newton sucks'
      stop
      end function dnewton




      function dkc_acos(x)
      real*8 A, B, C, D, E, F, x, x2, x4, x8, x16, x32
      real*8 dkc_acos, halfpi
      halfpi = 1.5707963267949

      A =  0.445652942766293691
      B = -0.816532368071267398
      C =  1.146354678043796890
      D = -0.866198697440988434
      E =  0.652623805576433744
      F =  1.003400074598418900

      x2 = x*x
      x4 = x2*x2
      x8 = x4*x4
      x16 = x8*x8
      x32 = x16*x16

      dkc_acos = ((A*x32 + B)*x32 + C*x16 + D*x8 + E*x4 + F) * x
      dkc_acos = halfpi - dkc_acos
      return
      end function dkc_acos

      function dkc15_acos(x)
      real*8 x, x2, x4, A, B
      real*8 dkc15_acos, halfpi
      halfpi = 1.5707963267949

      A = 0.622863420437093846
      B = 0.881157804380722130

      x2 = x*x
      x4 = x2*x2

      dkc15_acos = (A*x4 + B) * x
      dkc15_acos = halfpi - dkc15_acos
      return
      end function dkc15_acos





      FUNCTION ran2(idum)
      INTEGER idum,IM1,IM2,IMM1,IA1,IA2,IQ1,IQ2,IR1,IR2,NTAB,NDIV
      DOUBLE PRECISION ran2,AM,EPS,RNMX
      PARAMETER (IM1=2147483563,IM2=2147483399,AM=1./IM1,IMM1=IM1-1,
     +     IA1=40014,IA2=40692,IQ1=53668,IQ2=52774,IR1=12211,
     +     IR2=3791,NTAB=32,NDIV=1+IMM1/NTAB,EPS=1.2e-15,RNMX=1.-EPS)

c     Long period (> 2 x 10 18 ) random number generator of L'Ecuyer
c     with Bays-Durham shuffle and added safeguards. Returns a uniform
c     random deviate between 0.0 and 1.0 (exclusive of the endpoint
c     values). Call with idum a negative integer to initialize;
c     thereafter, do not alter idum between successive deviates in a
c     sequence. RNMX should approximate the largest floating value that
c     is less than 1.

      INTEGER idum2,j,k,iv(NTAB),iy
      SAVE iv,iy,idum2
      DATA idum2/123456789/, iv/NTAB*0/, iy/0/
      if (idum <= 0) then       !Initialize.
         idum=max(-idum,1)      !Be sure to prevent idum = 0.
         idum2=idum
         do j=NTAB+8,1,-1       !Load the shuffle table (after 8 warm-ups).
            k=idum/IQ1
            idum=IA1*(idum-k*IQ1)-k*IR1
            if (idum < 0) idum=idum+IM1
            if (j <= NTAB) iv(j)=idum
         enddo
         iy=iv(1)
      endif
      k=idum/IQ1                !Start here when not initializing.
      idum=IA1*(idum-k*IQ1)-k*IR1 !Compute idum=mod(IA1*idum,IM1) without over-
      if (idum < 0) idum=idum+IM1 !flows by Schrage's method.
      k=idum2/IQ2
      idum2=IA2*(idum2-k*IQ2)-k*IR2 !Compute idum2=mod(IA2*idum2,IM2) likewise.
      if (idum2 < 0) idum2=idum2+IM2
      j=1+iy/NDIV               !Will be in the range 1:NTAB.
      iy=iv(j)-idum2            !Here idum is shuffled, idum and idum2 are com-
      iv(j)=idum                !bined to generate output.
      if(iy < 1)iy=iy+IMM1
      ran2=dmin1(AM*dble(iy),RNMX) !Because users don't expect endpoint values.
      return
      END function ran2
