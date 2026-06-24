__global__ void KERNEL_Neumann_boundary3D_sph(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].p_type>0)	return;		// Immersed Boundary Method


	int_t icell,jcell,kcell;
	Real xi, yi, zi;
	Real uxi, uyi, uzi;
	Real rhoi, presi, tempi,pori;
	Real rho_ref_i;
	Real search_range, tmp_h, tmp_A;
	Real tflt_s;
	Real ttemp;
	Real tux, tuy, tuz;
	Real tpres, thpres, trho, aix, aiy, aiz;
	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(1.0*tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;
	rhoi=P1[i].rho;
	presi=P1[i].pres;
	tempi=P1[i].temp;
	pori=P1[i].DEMpor;

	tpres=thpres=tflt_s=ttemp=trho=0.0;
	tux=tuy=tuz=0.0;

	// 	// calculate I,J,K in cell
	// if((k_x_max==k_x_min)){icell=0;}
	// else{icell=min(floor((xi-k_x_min)/k_dcell),k_NI-1);}
	// if((k_y_max==k_y_min)){jcell=0;}
	// else{jcell=min(floor((yi-k_y_min)/k_dcell),k_NJ-1);}
	// if((k_z_max==k_z_min)){kcell=0;}
	// else{kcell=min(floor((zi-k_z_min)/k_dcell),k_NK-1);}
	// 	// out-of-range handling
	// 	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,zj,uxj,uyj,uzj,tdist,mj,rhoj,presj,tempj,porj;
					Real ajx,ajy,ajz;

					xj=P1[j].x;
					yj=P1[j].y;
					zj=P1[j].z;
					uxj=P1[j].ux;
					uyj=P1[j].uy;
					uzj=P1[j].uz;
					ajx=P3[j].ftotalx;
					ajy=P3[j].ftotaly;
					ajz=P3[j].ftotalz;
					mj=P1[j].m;
					rhoj=P1[j].rho;
					presj=P1[j].pres;
					tempj=P1[j].temp;
					porj=P1[j].DEMpor;

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
					if(P1[j].p_type<1000){

					if(tdist<2.0*search_range){
					// if(tdist<search_range){
						Real twij, tdwij, tdwx, tdwy, tdwz;
						int_t ptype_j=P1[j].p_type;
						int_t buffer_type_j=P1[j].buffer_type;
						twij=calc_kernel_wij(tmp_A,2.0*tmp_h,tdist);
						tdwij=calc_kernel_dwij(tmp_A,2.0*tmp_h,tdist);
						// twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
						// tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
						tdwx=(xi-xj)/tdist * tdwij;
						tdwy=(yi-yj)/tdist * tdwij;
						tdwz=(zi-zj)/tdist * tdwij;

						tpres+=mj/rhoj*presj*twij*(ptype_j>=1);
						thpres+=((P3[j].ftotalx)*(xj-xi)+(P3[j].ftotaly)*(yj-yi)+(P3[j].ftotalz)*(zj-zi))*mj/rhoj*twij*(ptype_j>=1);
						tux+=mj/rhoj*uxj*twij*(ptype_j>=1);
						tuy+=mj/rhoj*uyj*twij*(ptype_j>=1);
						tuz+=mj/rhoj*uzj*twij*(ptype_j>=1);
						tflt_s+=mj/rhoj*twij*(ptype_j>=1);
						ttemp+=mj/rhoj*tempj*twij*(ptype_j>=1);

						trho+=rhoj/porj*twij*mj/rhoj*(ptype_j>=1);

					}
				}
			}
				}
			}
		}
	}


	if(tflt_s<1e-6){
		P1[i].rho=P2[i].rho_ref;
		P1[i].pres=0.0;
	}else{
	P1[i].pres = (tpres)/tflt_s;
	P1[i].temp = ttemp/tflt_s;

	double B = 1.186*k_soundspeed*k_soundspeed/k_gamma;
	double K = P1[i].pres/B + 1.0;

	P1[i].rho=pori*trho/tflt_s;

	// P1[i].ux=tux/tflt_s;
	// P1[i].uy=tuy/tflt_s;
	// P1[i].uz=tuz/tflt_s;

	}

}

__global__ void KERNEL_open_boundary_extrapolation3D_1_sph(Real ttime, int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3,int_t tcount,Real tdt)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].buffer_type<=0) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method

	int_t icell,jcell,kcell;
	Real xi, xgnode_i, yi,ygnode_i, zi,zgnode_i;
	Real presi, rhoi, uxi, uyi, uzi, tempi;
	Real rho_ref_i;
	Real search_range, tmp_h, tmp_A, trho, tdrho_x, tdrho_y, tdrho_z, tux, tdux_x, tdux_y, tdux_z, tuy, tduy_x, tduy_y, tduy_z, tuz, tduz_x, tduz_y, tduz_z;
	Real drho_x, drho_y;
	Real dux_x, dux_y, duy_x, duy_y;
	Real tflt_s;
	Real tpres, tdpres_x, tdpres_y, tdpres_z, dpres_x, dpres_y;
	Real ttemp, tdtemp_x, tdtemp_y, tdtemp_z;
	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	Real p_ref, ux_ref, uy_ref, rho_ref;
	Real J1, J2, J3, J4;

	Real pori;

	pori=P1[i].DEMpor;
	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	presi=P1[i].pres;
	rhoi=P1[i].rho;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;
	tempi=P1[i].temp;

	// calculate ghost node

	if(P1[i].buffer_type==Inlet){
		xgnode_i=xi;
		ygnode_i=yi;
		zgnode_i=2*L2-zi;
	}else if(P1[i].buffer_type==Outlet){
		xgnode_i=xi;
		ygnode_i=yi;
		zgnode_i=2*L3-zi;
	}

	trho=tdrho_x=tdrho_y=tdrho_z=tux=tdux_x=tdux_y=tdux_z=tuy=tduy_x=tduy_y=tduy_z=tuz=tduz_x=tduz_y=tduz_z=tflt_s=tpres=tdpres_x=tdpres_y=tdpres_z=0.0;
	ttemp = tdtemp_x = tdtemp_y = tdtemp_z = 0;
	p_ref=0.0; uy_ref=1.0;  ux_ref=0.0; rho_ref=Inlet_Density;

	J1=J2=J3=J4=0.0; // characteristic wave


	// // calculate I,J,K in cell
	// if((k_x_max==k_x_min)){icell=0;}
	// else{icell=min(floor((xgnode_i-k_x_min)/k_dcell),k_NI-1);}
	// if((k_y_max==k_y_min)){jcell=0;}
	// else{jcell=min(floor((ygnode_i-k_y_min)/k_dcell),k_NJ-1);}
	// if((k_z_max==k_z_min)){kcell=0;}
	// else{kcell=min(floor((zgnode_i-k_z_min)/k_dcell),k_NK-1);}
	// // out-of-range handling
	// if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xgnode_i-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((ygnode_i-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zgnode_i-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
			int_t k=idx_cell(icell+x,jcell+y,kcell+z);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,zj,tdist,uxj,uyj,uzj,mj,rhoj,presj,tempj;
					Real volj,porj;

					xj=P1[j].x;
					yj=P1[j].y;
					zj=P1[j].z;

					mj=P1[j].m;
					rhoj=P1[j].rho;
					volj=P1[j].vol;
					uxj=P1[j].ux;
					uyj=P1[j].uy;
					uzj=P1[j].uz;
					presj=P1[j].pres;
					porj=P1[j].DEMpor;
					tempj=P1[j].temp;

					tdist=sqrt((xgnode_i-xj)*(xgnode_i-xj)+(ygnode_i-yj)*(ygnode_i-yj)+(zgnode_i-zj)*(zgnode_i-zj))+1e-20;
					if(P1[j].p_type<1000){

					if(tdist<search_range){
						Real twij, tdwij, tdwx, tdwy,tdwz;
						int_t ptype_j=P1[j].p_type;
						int_t buffer_type_j=P1[j].buffer_type;
						twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
						tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
						tdwx=(xgnode_i-xj)/tdist * tdwij;
						tdwy=(ygnode_i-yj)/tdist * tdwij;
						tdwz=(zgnode_i-zj)/tdist * tdwij;

						if(k_kgc_solve>0){
							apply_gradient_correction_3D(P3[i].Cm,twij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);
						}

						if (ptype_j==1 && buffer_type_j==0){

							trho+=rhoj/porj*twij*volj*(ptype_j==1);
							tdrho_x+=(rhoj/porj-rhoi/pori)*tdwx*volj*(ptype_j==1);
							tdrho_y+=(rhoj/porj-rhoi/pori)*tdwy*volj*(ptype_j==1);
							tdrho_z+=(rhoj/porj-rhoi/pori)*tdwz*volj*(ptype_j==1);
	
							tpres+=volj*presj*twij*(ptype_j==1);
							tdpres_x+=(presj-presi)*tdwx*volj*(ptype_j==1);
							tdpres_y+=(presj-presi)*tdwy*volj*(ptype_j==1);
							tdpres_z+=(presj-presi)*tdwz*volj*(ptype_j==1);
	
							tuz+=volj*uzj*twij*(ptype_j==1);
							tduz_x+=(uzj-uzi)*tdwx*volj*(ptype_j==1);
							tduz_y+=(uzj-uzi)*tdwy*volj*(ptype_j==1);
							tduz_z+=(uzj-uzi)*tdwz*volj*(ptype_j==1);

							tux+=volj*uxj*twij*(ptype_j==1);
							tdux_x+=(uxj-uxi)*tdwx*volj*(ptype_j==1);
							tdux_y+=(uxj-uxi)*tdwy*volj*(ptype_j==1);
							tdux_z+=(uxj-uxi)*tdwz*volj*(ptype_j==1);

							tuy+=volj*uyj*twij*(ptype_j==1);
							tduy_x+=(uyj-uyi)*tdwx*volj*(ptype_j==1);
							tduy_y+=(uyj-uyi)*tdwy*volj*(ptype_j==1);
							tduy_z+=(uyj-uyi)*tdwz*volj*(ptype_j==1);

							ttemp+=volj*tempj*twij*(ptype_j==1);
							tdtemp_x+=(tempj-tempi)*tdwx*volj*(ptype_j==1);
							tdtemp_y+=(tempj-tempi)*tdwy*volj*(ptype_j==1);
							tdtemp_z+=(tempj-tempi)*tdwz*volj*(ptype_j==1);

							tflt_s+=volj*twij*(ptype_j==1);

						}



					}
				}
				}
			}
		}
	}
	}

		if (k_open_boundary==1){

			if(P1[i].buffer_type==Inlet){	
					
				P1[i].OpenBC_pres = tpres/tflt_s + 0*(2*(zi-L2)) * tdpres_z;	//Pi = Pg + (zi - zg) * dPg/dz(corr_CSPM)
				//P1[i].rho = pori * (trho/tflt_s + 1*(2*(zi-L2)) * tdrho_z);	
				P1[i].OpenBC_rho = pori*DENSITY_AIR;
				P1[i].OpenBC_m = P1[i].rho * P1[i].vol0;	
				//P1[i].uz = 0.225/1.186;
				P1[i].OpenBC_uz = 24.045*2-tuz/tflt_s;
				P1[i].OpenBC_ux = 0-tux/tflt_s;			
				P1[i].OpenBC_uy = 0-tuy/tflt_s;	
				P1[i].OpenBC_temp = 823;	
				if((P1[i].elix<0.9999)&&(P1[i].p_type>0))	P1[i].OpenBC_m=P1[i].rho*P1[i].vol;
	
			}
			else if(P1[i].buffer_type==Outlet)
			{	

				P1[i].OpenBC_pres= 0.0;
				P1[i].OpenBC_rho = pori * (trho/tflt_s + 0*(2*(zi-L3)) * tdrho_z);	
				//P1[i].rho = 1.186;
				P1[i].OpenBC_m = P1[i].rho * P1[i].vol0;
				P1[i].OpenBC_uz = tuz/tflt_s + 0*(2*(zi-L3)) * tduz_z;
				P1[i].OpenBC_ux = tux/tflt_s;		
				P1[i].OpenBC_uy = tuy/tflt_s;
				P1[i].OpenBC_temp = ttemp/tflt_s + 0*(2*(zi-L3)) * tdtemp_z;
				//P1[i].temp = 200;	
				if((P1[i].elix<0.9999)&&(P1[i].p_type>0))	P1[i].OpenBC_m=P1[i].rho*P1[i].vol;
	
			}
		}
}

__global__ void KERNEL_open_boundary_extrapolation3D_1_calc_sph(Real ttime, int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3,int_t tcount,Real tdt)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].buffer_type<=0) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method

	P1[i].pres = P1[i].OpenBC_pres;
	P1[i].rho = P1[i].OpenBC_rho;
	P1[i].m = P1[i].OpenBC_m;
	P1[i].uz = P1[i].OpenBC_uz;
	P1[i].ux = P1[i].OpenBC_ux;
	P1[i].uy = P1[i].OpenBC_uy;
	P1[i].temp = P1[i].OpenBC_temp;
}