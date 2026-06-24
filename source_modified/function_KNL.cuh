__device__ Real calc_kernel_wij_ipf(Real tH,Real rr){

	Real tR,wij_ipf,tA;
	tR=wij_ipf=0.0;
	tA=1.0;
	Real eps;
	eps=tH/3.5;

	tR=rr/tH;
	wij_ipf=-(tR<2)*tA*cos(3*PI/4*tR);

	return wij_ipf;
}

__device__ Real calc_kernel_wij_half(Real tH,Real rr){

	Real tR,wij_half,tA;
	tR=wij_half=0.0;
	tA=1.0;
	Real eps0;
	eps0=tH/3.5*0.4;

	tR=rr/tH;
	wij_half=0.0;

	return wij_half;
}

__device__ Real calc_tmpA(Real tH)
{
	Real tA=0.0;
	//
	if(k_kernel_type==Gaussian){
		//if(k_dim==1) tA=1.0/(pow(PI,0.5)*tH);
		if(k_dim==2) tA=1.0/(PI*pow(tH,2));
		else if(k_dim==3) tA=1.0/(pow(PI,1.5)*pow(tH,3));
	}else	if(k_kernel_type==Quintic){
		//if(k_dim==1) tA=1.0;
		if(k_dim==2) tA=7.0/(478.0*PI*pow(tH,2));
		else if(k_dim==3) tA=3.0/(359.0*PI*pow(tH,3));
	}else	if(k_kernel_type==Quartic){
		//if(k_dim==1) tA=1.0/tH;
		if(k_dim==2) tA=15.0/(7.0*PI*pow(tH,2));
		else if(k_dim==3) tA=315.0/(208.0*PI*pow(tH,3));
	}else	if(k_kernel_type==Wendland2){
		//if(k_dim==1) tA=1.25/(2*tH); // 5.0/(4*(2h))
		if(k_dim==2) tA=2.228169203286535/(4*tH*tH);	// 7.0/(pi*(2h)^2)
		else if(k_dim==3) tA=3.342253804929802/(8*tH*tH*tH);	// 21.0/(2*pi*(2th)^3)
	}else	if(k_kernel_type==Wendland4){
		//if(k_dim==1) tA=1.5/(2*tH);	// 3.0/(2*(2h))
		if(k_dim==2) tA=2.864788975654116/(4*tH*tH);	// 9.0/(pi*(2h)^2)
		else if(k_dim==3) tA=4.923856051905513/(8*tH*tH*tH);	// 495.0/(32*pi*(2h)^3)
	}else	if(k_kernel_type==Wendland6){
		//if(k_dim==1) tA=1.71875/(2*tH);	// 55.0/(32*(2h))
		if(k_dim==2) tA=3.546881588905096/(4*tH*tH);	// 78.0/(7*pi*(2h)^2)
		else if(k_dim==3) tA=6.788953041263660/(8*tH*tH*tH);	// 1365.0/(64*pi*(2h)^3)
	}
	return tA;
}

__device__ Real calc_kernel_dwij(Real tA,Real tH,Real rr)
{
	Real tR,dwij;
	tR=dwij=0.0;

	if(k_kernel_type==Gaussian){
		tR=rr/tH;
		dwij=(1.0/tH)*tA*(-2.0)*tR*exp(-pow(tR,2));
	}else	if(k_kernel_type==Quintic){
		tR=rr/tH;
		if(tR<1) dwij=(1.0/tH)*tA*(pow(3.0-tR,5)+30.0*pow(2.0-tR,4)-75.0*pow(1.0-tR,4));
		else if(1<=tR&&tR<2) dwij=(1.0/tH)*tA*(-5.0*pow(3.0-tR,4)+30.0*pow(2.0-tR,4));
		else if(2<=tR&&tR<3) dwij=(1.0/tH)*tA*(-5.0*pow(3.0-tR,4));
	}else	if(k_kernel_type==Quartic){
		tR=rr/tH;
		dwij=(tR<2)*(1.0/tH)*tA*(-9.0/8.0*2*tR+19.0/24.0*3.0*pow(tR,2)-5.0/32.0*4.0*pow(tR,3));
	}else	if(k_kernel_type==Wendland2){
		tR=rr/tH*0.5;
		//if(k_dim==1) dwij=(tR<1)*(1/(2*tH))*tA*(-12*tR*(1-tR)*(1-tR));
		if(k_dim==2) dwij=(tR<1)*(1/(2*tH))*tA*(-20*tR*(1-tR)*(1-tR)*(1-tR));
		else if(k_dim==3) dwij=(tR<1)*(1/(2*tH))*tA*(-20*tR*(1-tR)*(1-tR)*(1-tR));
	}else	if(k_kernel_type==Wendland4){
		tR=rr/tH*0.5;
		//if(k_dim==1) dwij=(tR<1)*tA*(-14*tR*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+4*tR));
		if(k_dim==2) dwij=(tR<1)*(1/(2*tH))*tA*(-18.666666666666668*tR*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(5*tR+1));
		else if(k_dim==3) dwij=(tR<1)*(1/(2*tH))*tA*(-18.666666666666668*tR*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(5*tR+1));
	}else	if(k_kernel_type==Wendland6){
		tR=rr/tH*0.5;
		//if(k_dim==1) dwij=(tR<1)*(1/(2*tH))*tA*(-6*tR*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(35*tR*tR+18*tR+3));
		if(k_dim==2) dwij=(tR<1)*(1/(2*tH))*tA*(-22*tR*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(16*tR*tR+7*tR+1));
		else if(k_dim==3) dwij=(tR<1)*(1/(2*tH))*tA*(-22*tR*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(16*tR*tR+7*tR+1));
	}
	return dwij;
}

__device__ Real calc_kernel_wij(Real tA,Real tH,Real rr)
{

	Real tR,wij;
	tR=wij=0.0;

	if(k_kernel_type==Gaussian){
		tR=rr/tH;
		wij=tA*exp(-pow(tR,2));
	}else	if(k_kernel_type==Quintic){
		tR=rr/tH;
		if(tR<1) wij=tA*(pow(3.0-tR,5)-6.0*pow(2.0-tR,5)+15.0*pow(1.0-tR,5));
		else if(1<=tR&&tR<2) wij=tA*(pow(3.0-tR,5)-6.0*pow(2.0-tR,5));
		else if(2<=tR&&tR<3) wij=tA*(pow(3.0-tR,5));
	}else	if(k_kernel_type==Quartic){
		tR=rr/tH;
		wij=(tR<2)*tA*(2.0/3.0-9.0/8.0*pow(tR,2)+19.0/24.0*pow(tR,3)-5.0/32.0*pow(tR,4));
	}else	if(k_kernel_type==Wendland2){
		tR=rr/tH*0.5;
		//if(k_dim==1) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1+3*tR);
		if(k_dim==2) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+4*tR);
		else if(k_dim==3) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+4*tR);
	}else	if(k_kernel_type==Wendland4){
		tR=rr/tH*0.5;
		//if(k_dim==1) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+5*tR+8*tR*tR);
		if(k_dim==2) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+6*tR+11.666666666666666*tR*tR);
		else if(k_dim==3) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+6*tR+11.666666666666666*tR*tR);
	}else	if(k_kernel_type==Wendland6){
		tR=rr/tH*0.5;
		//if(k_dim==1) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+7*tR+19*tR*tR+21*tR*tR*tR);
		if(k_dim==2) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+8*tR+25*tR*tR+32*tR*tR*tR);
		else if(k_dim==3) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+8*tR+25*tR*tR+32*tR*tR*tR);
	}
	return wij;
}


__host__ __device__ void apply_gradient_correction_3D(Real Cm[][Correction_Matrix_Size],Real wij,Real dwx,Real dwy,Real dwz,Real* dwcx,Real* dwcy,Real* dwcz)
{
	if(k_kgc_solve==KGC) {
		*dwcx=Cm[0][0]*dwx + Cm[0][1]*dwy + Cm[0][2]*dwz;
		*dwcy=Cm[1][0]*dwx + Cm[1][1]*dwy + Cm[1][2]*dwz;
		*dwcz=Cm[2][0]*dwx + Cm[2][1]*dwy + Cm[2][2]*dwz;
	}else if(k_kgc_solve==FPM) {
		*dwcx=Cm[1][0]*wij + Cm[1][1]*dwx + Cm[1][2]*dwy + Cm[1][3]*dwz;
		*dwcy=Cm[2][0]*wij + Cm[2][1]*dwx + Cm[2][2]*dwy + Cm[2][3]*dwz;
		*dwcz=Cm[3][0]*wij + Cm[3][1]*dwx + Cm[3][2]*dwy + Cm[3][3]*dwz;
	}
	// else if(k_kgc_solve==DFPM) {
	// 	*dwcx=Cm[1][0]*wij + Cm[1][1]*dwx + Cm[1][2]*dwy + Cm[1][3]*dwz;
	// 	*dwcy=Cm[2][0]*wij + Cm[2][1]*dwx + Cm[2][2]*dwy + Cm[2][3]*dwz;
	// 	*dwcz=Cm[3][0]*wij + Cm[3][1]*dwx + Cm[3][2]*dwy + Cm[3][3]*dwz;
	// }else if(k_kgc_solve==KGF) {
	// 	*dwcx=Cm[1][0]*wij - Cm[1][1]*wij*(xi-xj) - Cm[1][2]*wij*(yi-yj) - Cm[1][3]*wij*(zi-zj);
	// 	*dwcy=Cm[2][0]*wij - Cm[2][1]*wij*(xi-xj) - Cm[2][2]*wij*(yi-yj) - Cm[2][3]*wij*(zi-zj);
	// 	*dwcz=Cm[3][0]*wij - Cm[3][1]*wij*(xi-xj) - Cm[3][2]*wij*(yi-yj) - Cm[3][3]*wij*(zi-zj);
	// }
}


