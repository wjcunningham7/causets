ccccccGENERATES NETWORKS OF THE CAUSAL STRUCTURE OF THE UNIVERSE WITH SCALING FUNCTION GIVEN BY
cccccccccccc   R(t)=xR0*(xOmegaM/xOmegaLambda)**(1./3.)*(Sinh[3t/2])^(-2/3)

      implicit double precision(x,r,d)
      character*80 filenameoutput
      parameter (NODOSMAX=10000000,NEDGESMAX=1500000)     !for larger graphs change these parameters


c%%%%%%%%%The volume element of the 3-sphere is dV=Sin^2(xtheta1) Sin(xtheta2)dtheta1 dtheta2 dphi

	  dimension xtheta1(1:NODOSMAX)                     !First angular coordinate of nodes in a 3-sphere. Runs between 0 and Pi
      dimension xtheta2(1:NODOSMAX)                     !Second angular coordinate of nodes in a 3-sphere. Runs between 0 and Pi
	  dimension xphi(1:NODOSMAX)                        !Third angular coordinate of nodes in a 3-sphere. Runs between 0 and 2Pi
	  dimension xeta(1:NODOSMAX)                        !conformal time of nodes
      dimension xcoordinate(1:4,1:2)                    !coordinates of the 3-D sphere in a 4D Euclidean space

c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      xtheta2gen(x)=dacos(1.-2.*x)

c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	  filenameoutput='Matter_Dark_Energy.net'                       !output filename
	  open(1,file=filenameoutput,status='unknown')

c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	  idum=-22                                                                !seed of the ran2 function
      xpi=3.141592653589793

c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   !model parameters
 	  xDelta=100.                                                               !density of events in space-time
      xtime=0.8458                                                              !observation time in units of a, i.e.,t/a
      NODOS=100000
      xalpha=(3.*dble(NODOS)/
     +(xpi**2*xDelta*(dsinh(3.*xtime)-3.*xtime)))**(1./3.)

c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	  xz=3.*xtime                                                              

      xetacurrent=2.*xf(xz/2.)/(3.*xalpha)
	  write(*,*) 'number of nodes=',NODOS,'eta current=',xetacurrent
c     pause
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%		

      do i=1,NODOS
      xphi(i)=2.*xpi*ran2(idum)                                               !assignment of theta to vertex i
	  xtheta1(i)=xtheta1gen(ran2(idum))
	  xtheta2(i)=xtheta2gen(ran2(idum))
	  xeta(i)=2.*xf(xzgen(ran2(idum),xz)/2.)/(3.*xalpha)             !assignment of conformal time to vertex i 
	  enddo
      write(*,*) 'done with assignments of coordinates'

c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c%%%%%%%%%%%%%%CONSTRUCTION OF THE NETWORK %%%%%%%%%%%%%%%%%%%%%%
      
      do i=1,NODOS-1
       call cpu_time(xtime_ini)
        do j=i+1,NODOS
		 xcoordinate(1,1)=dsin(xtheta1(i))*dsin(xtheta2(i))*dsin(xphi(i))
		 xcoordinate(2,1)=dsin(xtheta1(i))*dsin(xtheta2(i))*dcos(xphi(i))
		 xcoordinate(3,1)=dsin(xtheta1(i))*dcos(xtheta2(i))
		 xcoordinate(4,1)=dcos(xtheta1(i))
		 xcoordinate(1,2)=dsin(xtheta1(j))*dsin(xtheta2(j))*dsin(xphi(j))
		 xcoordinate(2,2)=dsin(xtheta1(j))*dsin(xtheta2(j))*dcos(xphi(j))
		 xcoordinate(3,2)=dsin(xtheta1(j))*dcos(xtheta2(j))
		 xcoordinate(4,2)=dcos(xtheta1(j))
		 
		 xprod=0.
		 do l=1,4
		  xprod=xprod+xcoordinate(l,1)*xcoordinate(l,2)
		 enddo
		  xangle_ij=dacos(xprod)
		  if(xangle_ij.gt.xpi) write(*,*) 'beware angle above pi!!'

		 xeta_i=xeta(i)
		 xeta_j=xeta(j)
		  if(xangle_ij.lt.dabs(xeta_i-xeta_j))then
			if(xeta_i.gt.xeta_j) write(1,100) j,i
			if(xeta_j.gt.xeta_i) write(1,100) i,j
          endif
        enddo
       call cpu_time(xtime_end)
	   write(*,*) 'process will end in',
     +  (xtime_end-xtime_ini)*dble(NODOS-i)/2.
	  enddo


	  close(1)
	  
      stop
100   format(I6,1x,I6)  
	  end




c%%%%%%%ALL FUNCTIONS START HERE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      FUNCTION xf(x)
        !This function computes the integral \int_0^x (Sinh[t])^(-2/3)dt
	  DOUBLE PRECISION xf,x,xuno
	    xuno=1.
		if(x.lt.1.)then
        xf=3.*x**(1./3.)-x**(7./3.)/21.+(4.*x**(13./3.))/1755.-
     +     (67.*x**(19./3.))/484785.+(11.*x**(25./3.))/1148175.-
     +     (2543.*x**(31./3.))/3523749075.+(593128.*x**(37./3.))/
     +     10333564666425.-(2561672.*x**(43./3.))/540417503500875.+
     +     (773684.*x**(49./3.))/1922880884549625.-(328376276.*
     +     x**(55./3.))/9415998076933231875.+(9610365492304.*
     +     x**(61./3.))/3127476681262988301121875.
	   else
           xf=2.**(2./3.)*(3.*(dexp(-2.*xuno/3.)-dexp(-2.*x/3.))/2.
     +                +2.*(dexp(-8.*xuno/3.)-dexp(-8.*x/3.))/8.
     +                +5.*(dexp(-14.*xuno/3.)-dexp(-14.*x/3.))/42.
     +                +6.*(dexp(-20.*xuno/3.)-dexp(-20.*x/3.))/81.
     +                +55.*(dexp(-26.*xuno/3.)-dexp(-26.*x/3.))/1053.)+
     +    3.*xuno**(1./3.)-xuno**(7./3.)/21.+(4.*xuno**(13./3.))/1755.-
     +     (67.*xuno**(19./3.))/484785.+(11.*xuno**(25./3.))/1148175.-
     +     (2543.*xuno**(31./3.))/3523749075.+(593128.*xuno**(37./3.))/
     +     10333564666425.-(2561672.*xuno**(43./3.))/540417503500875.+
     +     (773684.*xuno**(49./3.))/1922880884549625.-(328376276.*
     +     xuno**(55./3.))/9415998076933231875.+(9610365492304.*
     +     xuno**(61./3.))/3127476681262988301121875.
	   endif
	  return
      end

c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      FUNCTION xzgen(x,xz0)
        !this function generates random numbers for the quantity z=3t/a for a 
		!matter plus dark energy universe
	  implicit DOUBLE PRECISION (d)
	  DOUBLE PRECISION xzgen,x,xz0,xz_new,xz_old
	  logical ztrue
      PARAMETER (xprecision=1.e-14)
	  
	  xz_old=xz0
	  ztrue=.true.
	  do while(ztrue)
	   xz_new=xz_old+(x*dexp(xz0-xz_old)*(1.-dexp(-2.*xz0)-2.
     +*xz0*dexp(-xz0))+2.*xz_old*dexp(-xz_old)+
     + dexp(-2.*xz_old)-1.)/
     + (1.+dexp(-2.*xz_old)-2.*dexp(-xz_old))
       
	   if(dabs(xz_new-xz_old).lt.xprecision)then
	    xzgen=xz_new
		ztrue=.false.
	   endif	 

       if(xz_new.gt.xz_old)then
c	    write(*,*) 'loop found at',x
c		write(*,*) 'precision is',dabs(xz_new-xz_old)
        if(dabs(xz_new-xz_old).gt.xprecision) 
     +  write(*,*) 'loop reached. Best precision is',
     +  dabs(xz_new-xz_old)
        ztrue=.false.
		xzgen=xz_new
       endif
			
	   xz_old=xz_new
	  enddo
	  return
	  end


c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


      FUNCTION xtheta1gen(x)
           !this function generates random values of the first angular 
		   !coordinate of a node in the 3-sphere taking as input a random
		   !number x between 0 and 1
	  DOUBLE PRECISION xtheta1gen,x,xpi,xtheta_new,xtheta_old,xprecision
	  logical z
	  PARAMETER (xpi=3.141592653589793,xprecision=1.e-14)

	  xtheta_old=xpi/2.
	  z=.true.
	  do while(z)
	   xtheta_new=xtheta_old+(xpi*x-xtheta_old+0.5
     +*dsin(2.*xtheta_old))/(1.-dcos(2.*xtheta_old))
	   if(dabs(xtheta_new-xtheta_old).lt.xprecision)then
	    xtheta1gen=xtheta_new
		if(xtheta_new.lt.0.) write(*,*) 'beware theta below zero!'
		if(xtheta_new.gt.xpi) write(*,*) 'beware theta above pi!'
		z=.false.
	   endif	 
	   
	   
	   xtheta_old=xtheta_new
	  enddo
	  return
	  end

c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      FUNCTION ran2(idum)
      INTEGER idum,IM1,IM2,IMM1,IA1,IA2,IQ1,IQ2,IR1,IR2,NTAB,NDIV
      DOUBLE PRECISION ran2,AM,EPS,RNMX
      PARAMETER (IM1=2147483563,IM2=2147483399,AM=1./IM1,IMM1=IM1-1,
     +   IA1=40014,IA2=40692,IQ1=53668,IQ2=52774,IR1=12211,
     +   IR2=3791,NTAB=32,NDIV=1+IMM1/NTAB,EPS=1.2e-15,RNMX=1.-EPS)
        !Long period (> 2 x 10 18 ) random number generator of L'Ecuyer with Bays-Durham shuffle
        !and added safeguards. Returns a uniform random deviate between 0.0 and 1.0 (exclusive
        !of the endpoint values). Call with idum a negative integer to initialize; thereafter, do not
        !alter idum between successive deviates in a sequence. RNMX should approximate the largest
        !floating value that is less than 1.
      INTEGER idum2,j,k,iv(NTAB),iy
      SAVE iv,iy,idum2
      DATA idum2/123456789/, iv/NTAB*0/, iy/0/
      if (idum.le.0) then               !Initialize.
          idum=max(-idum,1)             !Be sure to prevent idum = 0.
          idum2=idum
          do j=NTAB+8,1,-1           !Load the shuffle table (after 8 warm-ups).
               k=idum/IQ1
               idum=IA1*(idum-k*IQ1)-k*IR1
               if (idum.lt.0) idum=idum+IM1
               if (j.le.NTAB) iv(j)=idum
          enddo 
          iy=iv(1)
      endif
      k=idum/IQ1                        !Start here when not initializing.
      idum=IA1*(idum-k*IQ1)-k*IR1       !Compute idum=mod(IA1*idum,IM1) without over-
      if (idum.lt.0) idum=idum+IM1      !flows by Schrage's method.
      k=idum2/IQ2
      idum2=IA2*(idum2-k*IQ2)-k*IR2     !Compute idum2=mod(IA2*idum2,IM2) likewise.
      if (idum2.lt.0) idum2=idum2+IM2
      j=1+iy/NDIV                       !Will be in the range 1:NTAB.
      iy=iv(j)-idum2                    !Here idum is shuffled, idum and idum2 are com-
      iv(j)=idum                        !bined to generate output.
      if(iy.lt.1)iy=iy+IMM1
      ran2=dmin1(AM*dble(iy),RNMX)              !Because users don't expect endpoint values.
      return
      END
