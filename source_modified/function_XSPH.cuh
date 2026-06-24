__global__ void KERNEL_xsph3D_sph(int_t inout,int_t*g_str,int_t*g_end,Real tdt,Real ttime,part1*P1,part2*P2)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type>1000) return;
	if((P1[i].p_type==0)||(P1[i].p_type==9)) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real uxi,uyi,uzi;
	Real tempi;
	Real tux,tuy,tuz,ttemp;																	// velocity
	Real tx0,ty0,tz0,xc,yc,zc;												// position ('0' : initial value/'c' : corrected value for Predictor-Corrector time stepping scheme)
	//Real flt_si;
	Real tmpx,tmpy,tmpz,tmptemp;
	Real search_range,tmp_h,tmp_A,tmp_flt;
	Real t_dt=tdt;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;								// search range

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	tempi=P1[i].temp;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;
	tx0=P2[i].x0;					// x-directional initial position
	ty0=P2[i].y0;					// x-directional initial position
	tz0=P2[i].z0;					// x-directional initial position
	//flt_si=P1[i].flt_s;
	tux=uxi;
	tuy=uyi;
	tuz=uzi;
	ttemp=tempi;

	//Real x_boundary=0.1*cos(PI*ttime)+3.79;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmpx=tmpy=tmpz=tmptemp=0.0;
	tmp_flt=0.0;
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						
						if((P1[j].p_type==1)||(P1[j].p_type==3)){

							Real xj,yj,zj,tdist;
							xj=P1[j].x;
							yj=P1[j].y;
							zj=P1[j].z;
	
							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
							if(tdist<search_range){
								Real twij,uxj,uyj,uzj,tempj,mj,rhoj;
								twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
								//p_type_j=p_type_[j];
								uxj=P1[j].ux;
								uyj=P1[j].uy;
								uzj=P1[j].uz;
								mj=P1[j].m;
								rhoj=P1[j].rho;
								tempj=P1[j].temp;
	
								tmpx+=k_c_xsph*mj/rhoj*(-uxi+uxj)*twij;
								tmpy+=k_c_xsph*mj/rhoj*(-uyi+uyj)*twij;
								tmpz+=k_c_xsph*mj/rhoj*(-uzi+uzj)*twij;

								tmptemp+=0.01*k_c_xsph*mj/rhoj*(tempj-tempi)*twij;

								tmp_flt+=mj/rhoj*twij;
							}



						}
						
						
					}
				}
			}
		}
	}

	xc=tx0+(tux+tmpx/tmp_flt)*(t_dt)*(ptypei>0);			// correct x-directional position
	yc=ty0+(tuy+tmpy/tmp_flt)*(t_dt)*(ptypei>0);			// correct Y-directional position
	zc=tz0+(tuz+tmpz/tmp_flt)*(t_dt)*(ptypei>0);			// correct Z-directional position

	P1[i].XSPH_ux=(tux+tmpx/tmp_flt)*(ptypei>0);				// update x-directional position
	P1[i].XSPH_uy=(tuy+tmpy/tmp_flt)*(ptypei>0);				// update y-directional position
	P1[i].XSPH_uz=(tuz+tmpz/tmp_flt)*(ptypei>0);				// update z-directional position

	P1[i].XSPH_temp=(ttemp+tmptemp/tmp_flt)*(ptypei>0);	
}

__global__ void KERNEL_xsph3D_calc_sph(int_t inout,int_t*g_str,int_t*g_end,Real tdt,Real ttime,part1*P1,part2*P2)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type>1000) return;
	if((P1[i].p_type==0)||(P1[i].p_type==9)) return;

	P1[i].ux=P1[i].XSPH_ux;				// update x-directional position
	P1[i].uy=P1[i].XSPH_uy;				// update y-directional position
	P1[i].uz=P1[i].XSPH_uz;
	P1[i].temp=P1[i].XSPH_temp;
}