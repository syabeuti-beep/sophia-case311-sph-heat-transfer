__global__ void KERNEL_advection_force3D_sph(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].i_type>i_type_crt) return;
	if(k_open_boundary>0 && P1[i].buffer_type>0) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	if(P1[i].p_type<=0)	return;		// Immersed Boundary Method

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi,uxi,uyi,uzi,kci,eta;
	Real pi,hi,mi,mi8,mri,rhoi,tempi,visi,betai;
	Real diffi,concni;
	Real nxi,nyi,nzi,nmagi,sigmai;			// for surface tension
	Real nx_ci,ny_ci,nz_ci,nmag_ci,curvi; 	// for surface tension
	Real search_range,tmp_A,tmp_Rc,tmp_Rd;
	Real tmpx,tmpy,tmpz,tmpn,tmpd;
	Real tmp_pgf_x,tmp_pgf_y,tmp_pgf_z;							// pressure gradient force term (for SPH-DEM Coupling)
	Real tmp_fsn, tmp_fsd;
	Real tmpsx, tmpsy, tmpsz;
	Real eulerx, eulery, eulerz, eulert;
	Real cpi, temp0;
	Real pori;
	// Thermal conduction with porosity (JHJ)
	Real epsi, one_minus_epsi;
	Real kfi, chii;
	// Thermal conduction with porosity (JHJ)

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;
	hi=P1[i].h;
	tempi=P1[i].temp;
	pi=P1[i].pres;
	mi=P1[i].m;
	rhoi=P1[i].rho;
	pori=P1[i].DEMpor;
	// Thermal conduction with porosity (JHJ)
	epsi = pori;
	// clamp epsilon
	if(epsi < (Real)1.0e-8) epsi = (Real)1.0e-8;
	if(epsi > (Real)1.0)    epsi = (Real)1.0;
	one_minus_epsi = (Real)1.0 - epsi;
	if(one_minus_epsi < (Real)0.0) one_minus_epsi = (Real)0.0;
	// Thermal conduction with porosity (JHJ)

	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range


	if(k_con_solve){
		// kci=conductivity(tempi,ptypei);
		// cpi=heat_capacity(tempi,ptypei);
		// eta=0.001*hi;
		// pure fluid thermal conductivity, k_f
		// Thermal conduction with porosity (JHJ)
		kfi = conductivity(tempi, ptypei);

		// k_f_eff = ((1 - sqrt(1 - epsilon)) / epsilon) * k_f
		kci = (((Real)1.0 - sqrt(one_minus_epsi)) / epsi) * kfi;

		// chi_i = epsilon_i * k_f_eff
		//       = (1 - sqrt(1 - epsilon_i)) * k_f
		chii = epsi * kci;

		cpi = heat_capacity(tempi, ptypei);

		// diffusion denominator regularization
		eta = (Real)0.1 * hi;
		// Thermal conduction with porosity (JHJ)
	}

	mi8=0.08/mi; // .. interface force
	mri=(mi/rhoi);

	visi=viscosity(tempi,ptypei)+P3[i].vis_t;
	betai=thermal_expansion(tempi,ptypei);

	// // calculate I,J,K in cell
	// if((k_x_max==k_x_min)){icell=0;}
	// else{icell=min(floor((xi-k_x_min)/k_dcell),k_NI-1);}
	// if((k_y_max==k_y_min)){jcell=0;}
	// else{jcell=min(floor((yi-k_y_min)/k_dcell),k_NJ-1);}
	// if((k_z_max==k_z_min)){kcell=0;}
	// else{kcell=min(floor((zi-k_z_min)/k_dcell),k_NK-1);}
	// // out-of-range handling
	// if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;


	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;



	tmpx=tmpy=tmpz=0.0;
	tmpn=0.0;
	tmpd=1.0;
	tmp_Rc=0.0;
	tmp_Rd=0.0;
	tmp_fsn=0.0;
	tmp_fsd=0.0;
	tmpsx=tmpsy=tmpsz=0.0;
	tmp_pgf_x=tmp_pgf_y=tmp_pgf_z=0.0;
	eulerx=eulery=eulerz=eulert=0.0;

	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,tdist;
						xj=P1[j].x;
						yj=P1[j].y;
						zj=P1[j].z;
						if(P1[j].p_type<1000){
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
						if(tdist<search_range){
							int_t ptypej;
							Real tdwx,tdwy,tdwz,uxj,uyj,uzj,mj,tempj,rhoj,pj,hj,kcj,sum_con_H,diffj,concnj,tmprd;
							Real nx_cj,ny_cj,nz_cj,nmag_cj,Phi_s,tmpnt;	// for surface tension

							Real twij=calc_kernel_wij(tmp_A,hi,tdist);
							Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);
							Real wij_ipf=calc_kernel_wij_ipf(hi,tdist);
							Real wij_half=calc_kernel_wij_half(hi,tdist);
							
							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
							tdwz=tdwij*(zi-zj)/tdist;


							// if(k_kgc_solve>0){
							apply_gradient_correction_3D(P3[i].Cm,twij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);
							// }

							ptypej=P1[j].p_type;
							uxj=P1[j].ux;
							uyj=P1[j].uy;
							uzj=P1[j].uz;
							mj=P1[j].m;
							tempj=P1[j].temp;
							rhoj=P1[j].rho;
							pj=P1[j].pres;
							hj=P1[j].h;

							if(k_fp_solve){

									Real C_p=-pori*(mj)*(pi+pj)/(rhoi*rhoj);
									

									tmp_pgf_x+=rhoi/pori*C_p*tdwx;			// force per unit volume
									tmp_pgf_y+=rhoi/pori*C_p*tdwy;
									tmp_pgf_z+=rhoi/pori*C_p*tdwz;
							}

							if(k_fv_solve){
								Real visj,C_v;
								visj=viscosity(tempj,ptypej)+P3[j].vis_t;
								//C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj))*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;
								C_v=(xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz;
								C_v*=(visi*visj)/(visi+visj);
								C_v*=4*pori*(mj/(rhoi*rhoj));
								C_v/=tdist;
								C_v/=tdist;

								//if(ptypej==0)	C_v=0.0;	//for free slip
								
								tmpx+=C_v*(uxi-uxj);
								tmpy+=C_v*(uyi-uyj);
								tmpz+=C_v*(uzi-uzj);
									
							}
							if(k_fva_solve){
								Real uij_xij=(uxi-uxj)*(xi-xj)+(uyi-uyj)*(yi-yj)+(uzi-uzj)*(zi-zj);
								if(uij_xij<0){
									Real h_ij,phi_ij,P_ij;
									//
									h_ij=(hi+hj)*0.5;
									phi_ij=h_ij*uij_xij;
									phi_ij/=(tdist*tdist+0.01*h_ij*h_ij);
									P_ij=mi*phi_ij*(-Alpha*k_soundspeed+Beta*phi_ij);
									P_ij/=(rhoi+rhoj);
									P_ij*=0.5;
									//P_ij=mi*(-Alpha*k_soundspeed*phi_ij+Beta*phi_ij*phi_ij)/(rhoi+rhoj)*0.5;
									tmpx+=-(P_ij)*tdwx;
									tmpy+=-(P_ij)*tdwy;
									tmpz+=-(P_ij)*tdwz;
								}
							}

							if(k_interface_solve){
								int_t flag;
								Real mrj,C_i;
								//
								flag=(ptypei!=BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej);
								mrj=mj/rhoj;
								C_i=abs(pi)*mri*mri+abs(pj)*mrj*mrj*(flag);
								C_i*=mi8*tdwij/tdist;

								//C_i=0.08/mi*(abs(pi)*(mi/rhoi)*(mi/rhoi)+abs(pj)*(mj/rhoj)*(mj/rhoj)*((ptypei!= BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej)))*tdwij/tdist;
								// apply interface sharpness force just for the fluid particles (2017.06.22 jyb)
								tmpx+=C_i*(xj-xi);
								tmpy+=C_i*(yj-yi);
								tmpz+=C_i*(zj-zi);
							}
							if(k_fb_solve){
								if((ptypei==FLUID)&(ptypej!=FLUID)){
									Real fb_ij=k_c_repulsive/(tdist+1e-10)/(tdist+1e-10)*twij*(2*mj/(mi+mj));

									tmpx+=fb_ij*(xi-xj);
									tmpy+=fb_ij*(yi-yj);
									tmpz+=fb_ij*(zi-zj);
								}
							}
							if(k_fs_solve){
								if(k_surf_model==2){

									nx_cj=P3[j].nx_c;
									ny_cj=P3[j].ny_c;
									nz_cj=P3[j].nz_c;
	
									nmag_cj=P3[j].nmag_c;
									Phi_s=-(ptypei!= ptypej)+(ptypei==ptypej);
	
									tmpnt=((nx_ci/nmag_ci)-Phi_s*(nx_cj/nmag_cj))*(xj-xi);
									tmpnt+=((ny_ci/nmag_ci)-Phi_s*(ny_cj/nmag_cj))*(yj-yi);
									tmpnt+=((nz_ci/nmag_ci)-Phi_s*(nz_cj/nmag_cj))*(zj-zi);
	
									tmpnt*=k_dim*(mj/rhoj)*tdwij/tdist;
									tmp_fsn+=tmpnt;
									tmp_fsd+=(mj/rhoj)*tdist*abs(tdwij);
									}else	if(k_surf_model==1){
										Real Cs;
									if ((ptypei == 1) & (ptypej == 1))
									{
										Cs=s_ff1 * (A_ff1 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_ff1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if ((ptypei == 2) & (ptypej == 2))
									{
										Cs=s_ff2 * (A_ff2 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_ff2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == 1) & (ptypej == 2)) || (ptypei == 2) & (ptypej == 1))
									{
										Cs=s_f1f2 * (A_f1f2 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_f1f2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == 0) & (ptypej == 1)) || (ptypei == 1) & (ptypej == 0))
									{
										Cs=s_sf1 * (A_sf1 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_sf1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == 0) & (ptypej == 2)) || (ptypei == 2) & (ptypej == 0))
									{
										Cs=s_sf2 * (A_sf2 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_sf2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == -1) & (ptypej == 1)) || (ptypei == 1) & (ptypej == -1))
									{
										Cs=s_s2f1 * (A_s2f1 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_s2f1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == -1) & (ptypej == 2)) || (ptypei == 2) & (ptypej == -1))
									{
										Cs=s_s2f2 * (A_s2f2 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_s2f2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else
									{
										Cs=0.0;
									}
									tmpsx=Cs*(xi - xj);
									tmpsy=Cs*(yi - yj);
									tmpsz=Cs*(zi - zj);
									tmpx+=Cs*(xi - xj);
									tmpy+=Cs*(yi - yj);
									tmpz+=Cs*(zi - zj);
	
									}
							}
							if((P1[i].elix<1.0)||(P1[i].eliy<1.0)){
								eulerx += (uxi*(uxj-uxi)*tdwx+uyi*(uxj-uxi)*tdwy+uzi*(uxj-uxi)*tdwz)*mj/rhoj*(1-P1[i].elix);
								eulery += (uxi*(uyj-uyi)*tdwx+uyi*(uyj-uyi)*tdwy+uzi*(uyj-uyi)*tdwz)*mj/rhoj*(1-P1[i].eliy);
								eulerz += (uxi*(uzj-uzi)*tdwx+uyi*(uzj-uzi)*tdwy+uzi*(uzj-uzi)*tdwz)*mj/rhoj*(1-P1[i].eliz);

								if(k_con_solve){
									eulert += (uxi*(tempj-tempi)*tdwx+uyi*(tempj-tempi)*tdwy+uzi*(tempj-tempi)*tdwz)*mj/rhoj*(1-P1[i].elix);
								}
							}
							if(k_con_solve){
								// kcj=conductivity(tempj, ptypej);
								// sum_con_H=4.0*pori*mj*kcj*kci*(tempi-tempj)*tdwij;
								// sum_con_H/=(tdist+1e-10)*rhoi*rhoj*(kci+kcj);
								// tmp_Rc+=sum_con_H;
								// Thermal conduction with porosity (JHJ)
								if(ptypej > 0 && ptypej < 1000){
									Real epsj;
									Real one_minus_epsj;
									Real kfj;
									Real chij;
									Real chih;
									Real rdotgradW;
									Real denom_diff;
									Real storage_rhoi;

									epsj = P1[j].DEMpor;

									// clamp epsilon_j
									if(epsj < (Real)1.0e-8) epsj = (Real)1.0e-8;
									if(epsj > (Real)1.0)    epsj = (Real)1.0;

									one_minus_epsj = (Real)1.0 - epsj;
									if(one_minus_epsj < (Real)0.0) one_minus_epsj = (Real)0.0;

									// pure fluid thermal conductivity of j
									kfj = conductivity(tempj, ptypej);

									// k_f_eff,j = ((1 - sqrt(1 - epsilon_j)) / epsilon_j) * k_f,j
									kcj = (((Real)1.0 - sqrt(one_minus_epsj)) / epsj) * kfj;

									// chi_j = epsilon_j * k_f_eff,j
									chij = epsj * kcj;

									// harmonic mean of chi = epsilon * k_eff
									chih = ((Real)2.0 * chii * chij) / (chii + chij + (Real)1.0e-30);

									// r_ij dot grad_i W_ij
									// tdwx, tdwy, tdwz are already gradient-corrected above.
									rdotgradW  = (xi - xj) * tdwx;
									rdotgradW += (yi - yj) * tdwy;
									rdotgradW += (zi - zj) * tdwz;

									denom_diff = tdist * tdist + eta * eta;

									// ------------------------------------------------------------
									// div( epsilon * k_eff * grad(T) )
									// ------------------------------------------------------------
									sum_con_H  = (Real)2.0 * (mj / rhoj) * chih;
									sum_con_H *= (tempi - tempj);
									sum_con_H *= rdotgradW / denom_diff;

									// ------------------------------------------------------------
									// Convert to temperature equation before Cp division:
									//
									// dT/dt = 1 / (epsilon_i * rho_f * Cp)
									//         * div(epsilon * k_eff * grad(T))
									//
									// Therefore tmp_Rc should contain:
									// 1 / (epsilon_i * rho_f)
									// * div(epsilon * k_eff * grad(T))
									// ------------------------------------------------------------

									// Case A: P1[i].rho is pure fluid density rho_f
									storage_rhoi = epsi * rhoi;

									// Case B: if P1[i].rho is already apparent density epsilon_i * rho_f,
									// use this instead:
									// storage_rhoi = rhoi;

									sum_con_H /= (storage_rhoi + (Real)1.0e-30);

									tmp_Rc += sum_con_H;
								}
								// Thermal conduction with porosity (JHJ)
							}
						}
					}
				}
			}
		}
	}
	}
	// z-directional gravitational force
	if(k_fg_solve) tmpz+=-Gravitational_CONST;
	if((k_boussinesq_solve)&(ptypei>0)&(ptypei<1000)) tmpz+=betai*Gravitational_CONST*(tempi-temp0);
	if((k_fs_solve)&&(k_surf_model==2)){
		if((nmagi>0.1/hi)&(tmp_fsn>0)) curvi=tmp_fsn/tmp_fsd;
		else curvi=0;

		tmpx+=sigmai*curvi*nxi/rhoi;
		tmpy+=sigmai*curvi*nyi/rhoi;
		tmpz+=sigmai*curvi*nzi/rhoi;
	}

	P3[i].ftotalx=1*(tmpx-eulerx);
	P3[i].ftotaly=1*(tmpy-eulery);
	P3[i].ftotalz=1*(tmpz-eulerz);

	if(k_con_solve){
		//P3[i].denthalpy=tmp_Rc*(ptypei!=-1);
		P3[i].dtemp=tmp_Rc/cpi*(ptypei!=-1)-eulert;
	}

	P3[i].fsx=tmpsx;
	P3[i].fsy=tmpsy;
	P3[i].fsz=tmpsz;

	P1[i].pgf_x=tmp_pgf_x;
	P1[i].pgf_y=tmp_pgf_y;
	P1[i].pgf_z=tmp_pgf_z;

	P3[i].ftotal=sqrt(tmpx*tmpx+tmpy*tmpy+tmpz*tmpz);

	if(k_con_solve)	P3[i].denthalpy=tmp_Rc;
	if(k_concn_solve) P3[i].dconcn=tmp_Rd;
	
}


__global__ void KERNEL_PPE3D_sph(Real tdt, int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].i_type>i_type_crt) return;
	if(k_open_boundary>0 && P1[i].buffer_type>0) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	if(P1[i].p_type<=0)	return;		// Immersed Boundary Method
	
	int_t icell,jcell,kcell;
	Real xi,yi,zi,uxi,uyi,uzi,rhoi;
	Real xistar,yistar,zistar,drhostar;
	Real search_range,hi,tmp_A;
	Real tmpx,tmpy,flt;
	Real bi, bix, biy,biz;
	Real Aij;
	Real AijPij;
	Real pori;
	Real dpori, dppori;

	hi=P1[i].h;
	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	xistar=P1[i].x_star;
	yistar=P1[i].y_star;
	zistar=P1[i].z_star;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;
	rhoi=P1[i].rho;
	pori=P1[i].DEMpor;
	dpori=P1[i].dDEMpor;
	dppori=P1[i].dDEMpor_prev;

	drhostar=0.0;
	bi = Aij = AijPij = 0.0;
	bix = 0.0;
	biy = 0.0;
	biz = 0.0;
	flt = 0.0;


	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;


	// // calculate I,J,K in cell
	// if((k_x_max==k_x_min)){icell=0;}
	// else{icell=min(floor((xi-k_x_min)/k_dcell),k_NI-1);}
	// if((k_y_max==k_y_min)){jcell=0;}
	// else{jcell=min(floor((yi-k_y_min)/k_dcell),k_NJ-1);}
	// if((k_z_max==k_z_min)){kcell=0;}
	// else{kcell=min(floor((zi-k_z_min)/k_dcell),k_NK-1);}
	// // out-of-range handling
	// if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	for(int_t z=-1;z<=1;z++){
	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,kcell+z);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,zj,uxj,uyj,uzj,tdist;
					Real xjstar,yjstar,zjstar,tdiststar;
					Real rhoj,mj,pj,rho_ref_j;
					Real volj;
					Real porj;
					int p_type_j;
					int itype;

					itype=P1[j].i_type;
					mj=P1[j].m;

					if(P1[j].p_type<1000){
						if(itype!=4){
					xj=P1[j].x;
					yj=P1[j].y;
					zj=P1[j].z;
					xjstar=P1[j].x_star;
					yjstar=P1[j].y_star;
					zjstar=P1[j].z_star;
					uxj=P1[j].ux;
					uyj=P1[j].uy;
					uzj=P1[j].uz;
					rhoj=P1[j].rho;
					rho_ref_j=P2[j].rho_ref;
					volj=P1[j].vol;
					pj=P1[j].pres;
					porj=P1[j].DEMpor;

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
					tdiststar=sqrt((xistar-xjstar)*(xistar-xjstar)+(yistar-yjstar)*(yistar-yjstar)+(zistar-zjstar)*(zistar-zjstar))+1e-20;

					if(tdist<search_range){
						Real twij=calc_kernel_wij(tmp_A,hi,tdist);
						Real twijstar=calc_kernel_wij(tmp_A,hi,tdiststar);
						Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);

						Real tdwx=tdwij*(xi-xj)/tdist;
						Real tdwy=tdwij*(yi-yj)/tdist;
						Real tdwz=tdwij*(zi-zj)/tdist;
						apply_gradient_correction_3D(P3[i].Cm,twij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);


						bix += mj/rhoj*(pori*uxi-porj*uxj)*tdwx;
						biy += mj/rhoj*(pori*uyi-porj*uyj)*tdwy;
						biz += mj/rhoj*(pori*uzi-porj*uzj)*tdwz;

						drhostar += (mj/rho_ref_j)*twijstar;
						flt += (mj/rhoj)*twij;


						Real c = -((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;
						//Aij += 2.0*mj/rhoj*c;
						Aij += 1*mj*((pori*rhoi+porj*rhoj)/(rhoj*rhoi))*c;
						// AijPij += 2.0*mj/rhoj*c*pj;
						AijPij += 1*mj*((pori*rhoi+porj*rhoj)/(rhoj*rhoi))*c*pj;	
					}
				}
			}
				}
			}
			}
		}
	}
	bi = DENSITY_AIR*(bix+biy+biz+2*dpori-dppori)/tdt;
	P1[i].PPE3 = bi;
	P1[i].PPE4 = AijPij;
	// bi = -rhoi*(1.0-drhostar/flt)/tdt/tdt;
	P1[i].PPE1 = rhoi*(bix+biy+biz)/tdt;
	//P1[i].PPE1 = 1;
	//P1[i].PPE2 = -rhoi*(1.0-drhostar/flt)/tdt/tdt;

	P1[i].PPE2 = Aij;

	// if(P3[i].lbl_surf==0)	P1[i].pres = (bi+AijPij)/Aij;
	// if(P3[i].lbl_surf!=0)	P1[i].pres = 0;
	// P1[i].pres = (bi+AijPij)/Aij;
	// if(P1[i].pres<0.0){
	// 	P1[i].pres=0.0;
	// }	
}

__global__ void KERNEL_PPE3D_calc_sph(Real tdt, int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].i_type>i_type_crt) return;
	if(k_open_boundary>0 && P1[i].buffer_type>0) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	if(P1[i].p_type<=0)	return;		// Immersed Boundary Method

	P1[i].pres = (P1[i].PPE3 + P1[i].PPE4)/P1[i].PPE2;
}

__global__ void KERNEL_pressureforce3D_sph(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
		uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
		if(i>=k_num_part2_sph) return;
		if(P1[i].i_type>i_type_crt) return;
		if(k_open_boundary>0 && P1[i].buffer_type>0) return;
		if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
		if(P1[i].p_type<=0)	return;		// Immersed Boundary Method
	
		int_t ptypei;
		int_t icell,jcell,kcell;
		Real xi,yi,zi,uxi,uyi,uzi,kci,eta;
		Real pi,hi,mi,mi8,mri,rhoi,tempi,visi,betai;
		Real diffi,concni;
		Real nxi,nyi,nzi,nmagi,sigmai;			// for surface tension
		Real nx_ci,ny_ci,nz_ci,nmag_ci,curvi; 	// for surface tension
		Real search_range,tmp_A,tmp_Rc,tmp_Rd;
		Real tmpx,tmpy,tmpz,tmpn,tmpd;
		Real tmp_fsn, tmp_fsd;
		Real tmppx,tmppy,tmppz;
		Real num;
		Real pori;
	
		ptypei=P1[i].p_type;
	
		xi=P1[i].x;
		yi=P1[i].y;
		zi=P1[i].z;
		uxi=P1[i].ux;
		uyi=P1[i].uy;
		uzi=P1[i].uz;
		hi=P1[i].h;
		tempi=P1[i].temp;
		pi=P1[i].pres;
		mi=P1[i].m;
		rhoi=P1[i].rho;
		pori=P1[i].DEMpor;
	
		tmp_A=calc_tmpA(hi);
		search_range=k_search_kappa*hi;	// search range
	

	
		if(k_con_solve){
			eta=0.001*hi;
			kci=conductivity(tempi,ptypei);
		}

	
		mi8=0.08/mi; // .. interface force
		mri=(mi/rhoi);
	
		visi=viscosity(tempi,ptypei)+P3[i].vis_t;
		betai=thermal_expansion(tempi,ptypei);
	
		// // calculate I,J,K in cell
		// if((k_x_max==k_x_min)){icell=0;}
		// else{icell=min(floor((xi-k_x_min)/k_dcell),k_NI-1);}
		// if((k_y_max==k_y_min)){jcell=0;}
		// else{jcell=min(floor((yi-k_y_min)/k_dcell),k_NJ-1);}
		// if((k_z_max==k_z_min)){kcell=0;}
		// else{kcell=min(floor((zi-k_z_min)/k_dcell),k_NK-1);}
		// // out-of-range handling
		// if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;


		// calculate I,J,K in cell
		if((k_x_max==k_x_min)){icell=0;}
		else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
		if((k_y_max==k_y_min)){jcell=0;}
		else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
		if((k_z_max==k_z_min)){kcell=0;}
		else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
		// out-of-range handling
		if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;	



		tmpx=tmpy=tmpz=0.0;
		tmpn=0.0;
		tmpd=1.0;
		tmp_Rc=0.0;
		tmp_Rd=0.0;
		tmp_fsn=0.0;
		tmp_fsd=0.0;
		tmppx=tmppy=tmppz=0.0;
		num=0.0;
	
		for(int_t z=-1;z<=1;z++){
			for(int_t y=-1;y<=1;y++){
				for(int_t x=-1;x<=1;x++){
					// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
					int_t k=idx_cell(icell+x,jcell+y,kcell+z);
	
					if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
					if(g_str[k]!=cu_memset){
						int_t fend=g_end[k];
						for(int_t j=g_str[k];j<fend;j++){
							Real xj,yj,zj,tdist;
							xj=P1[j].x;
							yj=P1[j].y;
							zj=P1[j].z;
							if(P1[j].p_type<1000){
							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
							if(tdist<search_range){
								int_t ptypej;
								Real tdwx,tdwy,tdwz,uxj,uyj,uzj,mj,tempj,rhoj,pj,hj,kcj,sum_con_H,diffj,concnj,tmprd;
								Real nx_cj,ny_cj,nz_cj,nmag_cj,Phi_s,tmpnt;	// for surface tension
	
								Real twij=calc_kernel_wij(tmp_A,hi,tdist);
								Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);
	
								tdwx=tdwij*(xi-xj)/tdist;
								tdwy=tdwij*(yi-yj)/tdist;
								tdwz=tdwij*(zi-zj)/tdist;
	
	
								// if(k_kgc_solve>0){
								apply_gradient_correction_3D(P3[i].Cm,twij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);
								// }
	
								ptypej=P1[j].p_type;
								uxj=P1[j].ux;
								uyj=P1[j].uy;
								uzj=P1[j].uz;
								mj=P1[j].m;
								tempj=P1[j].temp;
								rhoj=P1[j].rho;
								pj=P1[j].pres;
								hj=P1[j].h;
	
	
								if(k_fp_solve){
									Real C_p=-pori*1*mj*(pi+pj)/(rhoi*rhoj);
									//C_p=0.01;
									// tmpx+=C_p*tdwx;
									// tmpy+=C_p*tdwy;
									// tmpz+=C_p*tdwz;
									tmppx=C_p*tdwx;
									tmppy=C_p*tdwy;
									tmppz=C_p*tdwz;
									tmpx+=tmppx;
									tmpy+=tmppy;
									tmpz+=tmppz;

									num+=1;
									//tmpz+=1;
									//tmpz+=-mj*(pi+pj)/(rhoi*rhoj);
									//tmpz+=abs(tdwz);
								}
							}
						}
					}
			}
		}
		}
		}
		P3[i].fpx=tmpx;
		P3[i].fpy=tmpy;
		P3[i].fpz=tmpz;
		//P3[i].fpz=tmpz+100;
		//P1[i].test2=num;
		//printf("press : %f\n", P3[i].fpz);
	
}