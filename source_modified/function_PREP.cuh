__global__ void KERNEL_clc_correction_KGC_2D_sph(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell;
	Real search_range,tmp_h,tmp_A;
	Real xi,yi;
	Real tmpxx,tmpyy,tmpxy;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpxx=tmpyy=tmpxy=0;

	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			//int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);
			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){

					if(P1[j].p_type<=1000){
						Real xj,yj,tdist;
						xj=P1[j].x;
						yj=P1[j].y;

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
						if(tdist>0&&tdist<search_range){
							Real tdwij,mj,rhoj,txx,txy,tyy,rtd,mtd;
							tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
							mj=P1[j].m;
							rhoj=P1[j].rho;
							mtd=mj*tdwij;
							rtd=1.0/(rhoj*tdist);

							txx=-mtd*(xi-xj)*(xi-xj);
							txx*=rtd;
							txy=-mtd*(yi-yj)*(xi-xj);
							txy*=rtd;
							tyy=-mtd*(yi-yj)*(yi-yj);
							tyy*=rtd;

							tmpxx+=txx;
							tmpxy+=txy;
							tmpyy+=tyy;
						}

					}
					
				}
			}
		}
	}
	// save values to particle array

	Real tmpcmd=tmpxx*tmpyy-tmpxy*tmpxy;
	if(abs(tmpcmd)>Min_det){
		Real rtcmd=1.0/tmpcmd;
		P3[i].Cm[0][0]=tmpyy*rtcmd;
		P3[i].Cm[0][1]=-tmpxy*rtcmd;
		P3[i].Cm[1][0]=-tmpxy*rtcmd;
		P3[i].Cm[1][1]=tmpxx*rtcmd;
	}else{
		P3[i].Cm[0][0]=1;
		P3[i].Cm[0][1]=0;
		P3[i].Cm[1][0]=0;
		P3[i].Cm[1][1]=1;
	}
}

__global__ void KERNEL_clc_correction_KGC_3D_sph(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real search_range,tmp_h,tmp_A;;
	Real tmpxx,tmpyy,tmpzz,tmpxy,tmpyz,tmpzx;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmpxx=tmpyy=tmpzz=0;
	tmpxy=tmpyz=tmpzx=0;
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
					

						if(P1[j].p_type<=1000){

							Real xj,yj,zj,tdist;
							xj=P1[j].x;
							yj=P1[j].y;
							zj=P1[j].z;
	
							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
							if(tdist>0&&tdist<search_range){
								Real tdwij,mj,rhoj,txx,txy,tyy,tzx,tyz,tzz,rtd,mtd;
								tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
								mj=P1[j].m;
								rhoj=P1[j].rho;
	
								mtd=mj*tdwij;
								rtd=1.0/(rhoj*tdist);
	
								txx=-mtd*(xi-xj)*(xi-xj);
								txx*=rtd;
								txy=-mtd*(yi-yj)*(xi-xj);
								txy*=rtd;
								tyy=-mtd*(yi-yj)*(yi-yj);
								tyy*=rtd;
								tzx=-mtd*(xi-xj)*(zi-zj);
								tzx*=rtd;
								tyz=-mtd*(yi-yj)*(zi-zj);
								tyz*=rtd;
								tzz=-mtd*(zi-zj)*(zi-zj);
								tzz*=rtd;
								tmpxx+=txx;
								tmpxy+=txy;
								tmpyy+=tyy;
								tmpzx+=tzx;
								tmpyz+=tyz;
								tmpzz+=tzz;
							}

						}
						
					}
				}
			}
		}
	}
	// save values to particle array
	Real tmpcmd;
	tmpcmd=tmpxx*(tmpyy*tmpzz-tmpyz*tmpyz);
	tmpcmd-=tmpxy*(tmpxy*tmpzz-tmpyz*tmpzx);
	tmpcmd+=tmpzx*(tmpxy*tmpyz-tmpyy*tmpzx);

	if(abs(tmpcmd)>Min_det){
		Real rtcmd=1.0/tmpcmd;
		P3[i].Cm[0][0]=(tmpyy*tmpzz-tmpyz*tmpyz)*rtcmd;
		P3[i].Cm[0][1]=(tmpzx*tmpyz-tmpxy*tmpzz)*rtcmd;
		P3[i].Cm[0][2]=(tmpxy*tmpyz-tmpzx*tmpyy)*rtcmd;
		P3[i].Cm[1][0]=(tmpzx*tmpyz-tmpxy*tmpzz)*rtcmd;
		P3[i].Cm[1][1]=(tmpxx*tmpzz-tmpzx*tmpzx)*rtcmd;
		P3[i].Cm[1][2]=(tmpzx*tmpxy-tmpxx*tmpyz)*rtcmd;
		P3[i].Cm[2][0]=(tmpxy*tmpyz-tmpzx*tmpyy)*rtcmd;
		P3[i].Cm[2][1]=(tmpzx*tmpxy-tmpxx*tmpyz)*rtcmd;
		P3[i].Cm[2][2]=(tmpxx*tmpyy-tmpxy*tmpxy)*rtcmd;
	}
	else{
		P3[i].Cm[0][0]=1;
		P3[i].Cm[0][1]=0;
		P3[i].Cm[0][2]=0;
		P3[i].Cm[1][0]=0;
		P3[i].Cm[1][1]=1;
		P3[i].Cm[1][2]=0;
		P3[i].Cm[2][0]=0;
		P3[i].Cm[2][1]=0;
		P3[i].Cm[2][2]=1;
	}
}

void gradient_correction_sph(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	dim3 b,t;
	t.x=128;
	b.x=(num_part2_sph-1)/t.x+1;
	switch (kgc_solve){
		case KGC:
				if(dim==2) KERNEL_clc_correction_KGC_2D_sph<<<b,t>>>(g_str,g_end,P1,P3);
				if(dim==3) KERNEL_clc_correction_KGC_3D_sph<<<b,t>>>(g_str,g_end,P1,P3);
				cudaDeviceSynchronize();
				break;
		// case FPM:
		// 		if(dim==2) KERNEL_clc_correction_FPM_2D<<<b,t>>>(g_str,g_end,P1,P3);
		// 		// if(dim==3) KERNEL_clc_correction_FPM_3D<<<b,t>>>(inout,g_str,g_end,P1,P3);
		// 		// cudaDeviceSynchronize();
		// 		break;
		// case DFPM:
		// 		if(dim==2) KERNEL_clc_correction_DFPM_2D<<<b,t>>>(g_str,g_end,P1,P3);
		// 		// if(dim==3) KERNEL_clc_correction_DFPM_3D<<<b,t>>>(inout,g_str,g_end,P1,P3);
		// 		// cudaDeviceSynchronize();
		// 		break;
		// case KGF:
		// 		if(dim==2) KERNEL_clc_correction_KGF_2D<<<b,t>>>(g_str,g_end,P1,P3);
		// 		// if(dim==3) KERNEL_clc_correction_KGF_3D<<<b,t>>>(inout,g_str,g_end,P1,P3);
		// 		// cudaDeviceSynchronize();
		// 		break;
		default:
				if(dim==2) KERNEL_clc_correction_KGC_2D_sph<<<b,t>>>(g_str,g_end,P1,P3);
				if(dim==3) KERNEL_clc_correction_KGC_3D_sph<<<b,t>>>(g_str,g_end,P1,P3);
				cudaDeviceSynchronize();
				break;
	}
}

__global__ void KERNEL_clc_prep3D_coupling_sph(int_t*g_str_sph,int_t*g_end_sph,part1*P1_sph, part2*P2_sph, 
												int_t*g_str_dem,int_t*g_end_dem,part1*P1_dem, part2*P2_dem, 
												int_t tcount, Real ttime)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1_sph[i].i_type>i_type_crt) return;

	int_t icell,jcell,kcell;
	int_t ptypei=P1_sph[i].p_type;

	Real xi,yi,zi;
	Real mi=P1_sph[i].m;
	Real rhoi=P1_sph[i].rho;
	Real tmp_h,tmp_A,search_range;
	Real tmp_SPHflt, tmp_DEMflt,tmp_DEMfltd,tmp_por;				//tmp_filter, tmp_porosity(DEM calculation)
	Real tmp_DEMfltd_2;


	xi=P1_sph[i].x;
	yi=P1_sph[i].y;
	zi=P1_sph[i].z;


	tmp_h=1*P1_sph[i].h;
	tmp_A=calc_tmpA(1.0*tmp_h);
	
	search_range=k_search_kappa*1.0*tmp_h;	// search range


	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	// 초기화
	tmp_por=0.0;
	tmp_SPHflt=0.0;
	tmp_DEMflt=0.0;
	tmp_DEMfltd=0.0;
	tmp_DEMfltd_2=0.0;
	
	// sph-sph 계산
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

			//	if(k<0||k>=k_num_cells-1) continue;
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str_sph[k]!=cu_memset){
					int_t fend=g_end_sph[k];
					for(int_t j=g_str_sph[k];j<fend;j++){

						Real tmp_wij,tdist;
						Real xj,yj,zj;


						xj=P1_sph[j].x;
						yj=P1_sph[j].y;
						zj=P1_sph[j].z;

						
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
	
						if(tdist<search_range){

							tmp_wij=calc_kernel_wij(tmp_A,1.0*tmp_h,tdist);


							Real mj,rhoj,radj;
							int_t ptypej;

							mj=P1_sph[j].m;
							rhoj=P1_sph[j].rho;
							radj=P1_sph[j].rad;
							ptypej=P1_sph[j].p_type;

							if (ptypei<=1000){
								if((ptypej<=1000)){
								//if((ptypej<=1000)){
									tmp_SPHflt+=mj/rhoj*tmp_wij*(ptypej==1);
									//tmp_SPHflt+=mj/rhoj*1*(ptypej==1);	//simmple summation
								}
							}
						}	
					}
				}
			}
		}
	}


	// dem-dem 계산
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

			//	if(k<0||k>=k_num_cells-1) continue;
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str_dem[k]!=cu_memset){
					int_t fend=g_end_dem[k];
					for(int_t j=g_str_dem[k];j<fend;j++){

						Real tmp_wij,tdist;
						Real xj,yj,zj;


						xj=P1_dem[j].x;
						yj=P1_dem[j].y;
						zj=P1_dem[j].z;

						
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
	
						if(tdist<search_range){

							tmp_wij=calc_kernel_wij(tmp_A,1.0*tmp_h,tdist);


							Real mj,rhoj,radj;
							int_t ptypej;

							mj=P1_dem[j].m;
							rhoj=P1_dem[j].rho;
							radj=P1_dem[j].rad;
							ptypej=P1_dem[j].p_type;

							if (ptypei<=1000){
								if(ptypej>1000){

									tmp_por+=tmp_wij*4.18879*radj*radj*radj;
									//tmp_por+=1*4.18879*radj*radj*radj;	//simple summation

								}	  
							}
						}	
					}
				}
			}
		}
	}
	if(ptypei<=1000) {
		//P1[i].DEMpor=1.0-(zi<0.186)*(zi>0.00)*(tmp_por/(tmp_SPHflt+1e-20));
		P1_sph[i].DEMpor=1.0-(tmp_por/(tmp_SPHflt+1e-20));
	}
}

__global__ void KERNEL_clc_prep3D_coupling_dem(int_t*g_str_dem,int_t*g_end_dem,part1*P1_dem, part2*P2_dem,
												int_t*g_str_sph,int_t*g_end_sph,part1*P1_sph, part2*P2_sph,
												int_t tcount, Real ttime)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_dem) return;
	if(P1_dem[i].i_type>i_type_crt) return;

	int_t icell,jcell,kcell;
	int_t ptypei=P1_dem[i].p_type;

	Real xi,yi,zi;
	Real mi=P1_dem[i].m;
	Real rhoi=P1_dem[i].rho;
	Real tmp_h,tmp_A,search_range;
	Real tmp_SPHflt, tmp_DEMflt,tmp_DEMfltd,tmp_por;				//tmp_filter, tmp_porosity(DEM calculation)
	Real tmp_DEMfltd_2;


	xi=P1_dem[i].x;
	yi=P1_dem[i].y;
	zi=P1_dem[i].z;


	tmp_h=1*P1_dem[i].h;
	tmp_A=calc_tmpA(1.0*tmp_h);
	
	search_range=k_search_kappa*1.0*tmp_h;	// search range


	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	// 초기화
	tmp_por=0.0;
	tmp_SPHflt=0.0;
	tmp_DEMflt=0.0;
	tmp_DEMfltd=0.0;
	tmp_DEMfltd_2=0.0;
	
	// dem-sph 계산
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

			//	if(k<0||k>=k_num_cells-1) continue;
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str_sph[k]!=cu_memset){
					int_t fend=g_end_sph[k];
					for(int_t j=g_str_sph[k];j<fend;j++){

						Real tmp_wij,tdist;
						Real xj,yj,zj;


						xj=P1_sph[j].x;
						yj=P1_sph[j].y;
						zj=P1_sph[j].z;

						
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
	
						if(tdist<search_range){

							tmp_wij=calc_kernel_wij(tmp_A,1.0*tmp_h,tdist);


							Real mj,rhoj,radj;
							int_t ptypej;

							mj=P1_sph[j].m;
							rhoj=P1_sph[j].rho;
							radj=P1_sph[j].rad;
							ptypej=P1_sph[j].p_type;

							if(ptypei>1000){
								//if(ptypej<=1000){
								if((ptypej<=1000) && (ptypej!=0) && (ptypej!=MOVING) && (ptypej==1)){
									tmp_DEMfltd+=mj/rhoj*tmp_wij;
									tmp_DEMfltd_2+=mj/rhoj*tmp_wij;
								}	
								if((ptypej<=1000) && (ptypej!=0) && (ptypej!=MOVING) && (ptypej==3)){
									tmp_DEMflt+=mj/rhoj*tmp_wij;
								}	
							}
						}	
					}
				}
			}
		}
	}

	// dem-dem 계산
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

			//	if(k<0||k>=k_num_cells-1) continue;
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str_dem[k]!=cu_memset){
					int_t fend=g_end_dem[k];
					for(int_t j=g_str_dem[k];j<fend;j++){

						Real tmp_wij,tdist;
						Real xj,yj,zj;


						xj=P1_dem[j].x;
						yj=P1_dem[j].y;
						zj=P1_dem[j].z;

						
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
	
						if(tdist<search_range){

							tmp_wij=calc_kernel_wij(tmp_A,1.0*tmp_h,tdist);


							Real mj,rhoj,radj;
							int_t ptypej;

							mj=P1_dem[j].m;
							rhoj=P1_dem[j].rho;
							radj=P1_dem[j].rad;
							ptypej=P1_dem[j].p_type;

							if(ptypei>1000){
								if(ptypej>1000) tmp_por+=tmp_wij*4.18879*radj*radj*radj;
							}
						}	
					}
				}
			}
		}
	}


	// filter
	if(ptypei>1000){
		P1_dem[i].flt_s=tmp_DEMflt;
		P1_dem[i].flt_sd=tmp_DEMfltd;
		P1_dem[i].flt_sd_2=tmp_DEMfltd_2;
		P1_dem[i].DEMpor=1.0-(tmp_por/(tmp_DEMflt+tmp_DEMfltd+1.0e-20));
	}
}


__global__ void KERNEL_clc_prep3D_prep_sph(int_t*g_str,int_t*g_end,part1*P1, part2*P2, part3*P3, int_t tcount)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell,kcell;
	int_t ptypei=P1[i].p_type;

	Real xi,yi,zi;
	Real uxi,uyi,uzi;
	
	Real mi=P1[i].m;
	Real rhoi=P1[i].rho;
	Real pori=P1[i].DEMpor;
	Real cci=P3[i].cc;
	Real tmp_h,tmp_A,search_range;
	Real tmp_flt,tmp_SR;
	Real tmp_rhox,tmp_rhoy,tmp_rhoz;
	Real tmp_ncx, tmp_ncy, tmp_ncz, tmp_nx, tmp_ny, tmp_nz;
	Real tvis_t=0.0,th;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;

	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	th=tmp_h*L_SPS;
	search_range=k_search_kappa*tmp_h;	// search range

	Real m_ref;

	if (ptypei==-3)	m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	else if (ptypei==9) m_ref=DENSITY_WATER*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	else if (ptypei==0) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;	
	else if (ptypei==3) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	else if (ptypei==1) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	else m_ref=DENSITY_WATER*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;

	if ((ptypei==9)&&(zi<0.05)) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	
	// reference density
	if(k_dim==2) P2[i].rho_ref=m_ref/((tmp_h/1.600)*(tmp_h/1.600));
	//if(k_dim==3) P2[i].rho_ref=mi/((tmp_h/1.600)*(tmp_h/1.600)*(tmp_h/1.600));
	if(k_dim==3) P2[i].rho_ref=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
	if((k_rho_type==Continuity)){
		P2[i].rho0=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
		P1[i].rho=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
		P1[i].m=m_ref;
	} 
}

__global__ void KERNEL_clc_prep3D_sph(int_t*g_str,int_t*g_end,part1*P1, part2*P2, part3*P3, int_t tcount)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell,kcell;
	int_t ptypei=P1[i].p_type;

	Real xi,yi,zi;
	Real uxi,uyi,uzi;
	
	Real mi=P1[i].m;
	Real rhoi=P1[i].rho;
	Real pori=P1[i].DEMpor;
	Real cci=P3[i].cc;
	Real tmp_h,tmp_A,search_range;
	Real tmp_flt,tmp_SR;
	Real tmp_rhox,tmp_rhoy,tmp_rhoz;
	Real tmp_ncx, tmp_ncy, tmp_ncz, tmp_nx, tmp_ny, tmp_nz;
	Real tvis_t=0.0,th;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;

	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	th=tmp_h*L_SPS;
	search_range=k_search_kappa*tmp_h;	// search range

	Real m_ref;

	// if (ptypei==-3)	m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	// else if (ptypei==9) m_ref=DENSITY_WATER*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	// else if (ptypei==0) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;	
	// else if (ptypei==3) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	// else if (ptypei==1) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	// else m_ref=DENSITY_WATER*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;

	// if ((ptypei==9)&&(zi<0.05)) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	
	// // reference density
	// if(k_dim==2) P2[i].rho_ref=m_ref/((tmp_h/1.600)*(tmp_h/1.600));
	// //if(k_dim==3) P2[i].rho_ref=mi/((tmp_h/1.600)*(tmp_h/1.600)*(tmp_h/1.600));
	// if(k_dim==3) P2[i].rho_ref=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
	// if((k_rho_type==Continuity)){
	// 	P2[i].rho0=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
	// 	P1[i].rho=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
	// 	P1[i].m=m_ref;
	// } 

	// __syncthreads();

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	// 초기화
	tmp_flt=tmp_SR=tmp_rhox=tmp_rhoy=tmp_rhoz=0.0;
	tmp_nx=tmp_ny=tmp_nz=tmp_ncx=tmp_ncy=tmp_ncz=0.0;

	// 계산
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

			//	if(k<0||k>=k_num_cells-1) continue;
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						
						if(P1[j].p_type<=1000){
							Real xj,yj,zj,uxj,uyj,uzj,uij2,mj,rhoj,rho_refj,ccj, tdwx,tdwy,tdwz,tmp_wij,tmp_dwij,tdist,tmp_val;
							int_t ptypej;
	
							xj=P1[j].x;
							yj=P1[j].y;
							zj=P1[j].z;
							uxj=P1[j].ux;
							uyj=P1[j].uy;
							uzj=P1[j].uz;
							mj=P1[j].m;
							rhoj=P1[j].rho;
							rho_refj=P2[j].rho_ref;
							ptypej=P1[j].p_type;
							ccj=P3[j].cc;
	
							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
	
							if(tdist<search_range){
								tmp_wij=calc_kernel_wij(tmp_A,tmp_h,tdist);
								tmp_dwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
	
								tdwx=tmp_dwij*(xi-xj)/tdist;
								tdwy=tmp_dwij*(yi-yj)/tdist;
								tdwz=tmp_dwij*(zi-zj)/tdist;
	
								// filter
								if((tcount%(k_freq_filt*k_decouple_stride))==0){

									// if ((ptypej==0)|(ptypej==9)){
									// 	tmp_flt+=mj/DENSITY_WATER*tmp_wij;
									// }
									// else{
										tmp_flt+=mj/rhoj*tmp_wij;
									// }


								} 
								// filter
								//if((tcount%(k_freq_filt*k_decouple_stride))==0) tmp_flt+=(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*tmp_wij;
	
								// strain rate
								if((k_fv_solve==1)&&(k_turbulence_model!=Laminar))
								{
									uij2=(uxi-uxj)*(uxi-uxj)+(uyi-uyj)*(uyi-uyj)+(uzi-uzj)*(uzi-uzj);
									tmp_val=-0.5*mj*(rhoi+rhoj)*uij2;
									tmp_val/=(rhoi*rhoj*tdist*tdist);
									tmp_SR+=tmp_val*(xi-xj)*tdwx+tmp_val*(yi-yj)*tdwy+tmp_val*(zi-zj)*tdwz;
								}
	
								// gradient rho (for delta-sph)
								if(k_delSPH_solve==Antuono)
								{
									apply_gradient_correction_3D(P3[i].Cm,tmp_wij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);

									Real rhoi_ref=P2[i].rho_ref;
									Real rhoj_ref=P2[j].rho_ref;
		
									tmp_rhox+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwx;
									tmp_rhoy+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwy;
									tmp_rhoz+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwz;
								}
	
								// normal gradient for curvature
								if(k_fs_solve)
								{
									int wall;
									wall=1;
									tmp_ncx+=-(mj/rhoj)*(ccj-cci)*wall*(ptypei==ptypej)*tdwx;
									tmp_ncy+=-(mj/rhoj)*(ccj-cci)*wall*(ptypei==ptypej)*tdwy;
									tmp_ncz+=-(mj/rhoj)*(ccj-cci)*wall*(ptypei==ptypej)*tdwz;
	
									Real nC_s,nC_sx,nC_sy,nC_sz,nC_st;
	
									nC_s=(ptypei!=ptypej);
									nC_st=nC_s*((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj));
									nC_st*=(rhoi/(rhoi+rhoj))*(rhoi/mi)*tmp_dwij;
	
									nC_sx=nC_st*(xj-xi)/tdist;
									nC_sy=nC_st*(yj-yi)/tdist;
									nC_sz=nC_st*(zj-zi)/tdist;
	
									tmp_nx+=nC_sx;
									tmp_ny+=nC_sy;
									tmp_nz+=nC_sz;
								}
							}
						}
					}
				}
			}
		}
	}

	// strain_rate
	if((k_fv_solve==1)&&(k_turbulence_model!=Laminar)) {
		tmp_SR=max(1e-20,tmp_SR);
		P2[i].SR=sqrt(tmp_SR);

		if(k_turbulence_model==SPS) tvis_t=(Cs_SPS*th)*(Cs_SPS*th)*tmp_SR;
		P3[i].vis_t=tvis_t*rhoi;
	}

	// filter
	if((tcount%(k_freq_filt*k_decouple_stride))==0) P1[i].flt_s=tmp_flt;

	// gradient density
	if(k_delSPH_solve==Antuono){
		P1[i].grad_rhox=tmp_rhox;
		P1[i].grad_rhoy=tmp_rhoy;
		P1[i].grad_rhoz=tmp_rhoz;

		P1[i].test1=tmp_rhoy;
		P1[i].test5=tmp_rhoz;

	}

	// if((k_rho_type==Continuity) && (tcount==0)){
	// 	P2[i].rho0=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
	// 	P1[i].rho=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
	// } 

	// P1[i].m=P1[i].rho*P1[i].vol;

	//if(k_dim==3) P2[i].rho_ref=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));

	// normal gradient for surface tension
	if(k_fs_solve==1){
		Real tmpnmg=sqrt(tmp_ncx*tmp_ncx+tmp_ncy*tmp_ncy+tmp_ncz*tmp_ncz);

		P3[i].nx_c=tmp_ncx;
		P3[i].ny_c=tmp_ncy;
		P3[i].nz_c=tmp_ncz;

		P3[i].nmag_c=tmpnmg;
		if(tmpnmg<NORMAL_THRESHOLD){
			P3[i].nx_c=0;
			P3[i].ny_c=0;
			P3[i].nz_c=0;
			P3[i].nmag_c=1e-20;
		}

		// KERNEL_clc_normal_gradient3D ---------------
		P3[i].nx=tmp_nx;
		P3[i].ny=tmp_ny;
		P3[i].nz=tmp_nz;

		Real ntmpnmg=sqrt(tmp_nx*tmp_nx+tmp_ny*tmp_ny+tmp_nz*tmp_nz);
		P3[i].nmag=ntmpnmg;
		if(ntmpnmg<NORMAL_THRESHOLD){
			P3[i].nx=0;
			P3[i].ny=0;
			P3[i].nz=0;
			P3[i].nmag=1e-20;
		}
	}
}