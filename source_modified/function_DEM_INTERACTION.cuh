
__global__ void KERNEL_DEM_interaction3D_dem(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part1*TP1,part3*P3,Real ttime)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_dem) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;

	
	int_t ptypei, demidxi;
	int_t icell,jcell,kcell;
	Real xi,yi,zi,uxi,uyi,uzi,wxi,wyi,wzi;	// w : angular velocity in each direction 
	Real mi,mri,rhoi,rii;					// rii : rotational index for particle i
	Real radi, ymi, pri, Gi;					// DEM Calculation
	
	//Real tempi, kci, cpi, emi; - ORIGINAL
	Real tempi, temp1i, temp2i, temp3i, kci, kc1_i, kc2_i, kc3_i, kc0_i, alph_i, alph0_i, alph1_i, alph2_i, alph3_i, diff_kc0_i, diff_kc1_i, diff_kc2_i,diff_kc3_i, cpi, emi, CTE;

	//Real search_range;
	//Real tmpx,tmpy,tmpz, tmptqx,tmptqy,tmptqz,tmpn,tmpd,tmpdt;	// tmptq : temporary value for torque calculation -ORIGINAL
	Real tmpx,tmpy,tmpz,tmptqx,tmptqy,tmptqz,tmpn,tmpd,tmpdt,tmpdt1,tmpdt2,tmpdt3,tmpdt_rad,tmpdt_con,temp_avg;



	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;

	// velocity
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;

	// angular velocity
	wxi=P1[i].wx;
	wyi=P1[i].wy;
	wzi=P1[i].wz;

	radi=P1[i].rad;

	//tempi=P1[i].temp; - ORIGINAL
	tempi=P1[i].temp;
	temp1i=P1[i].temp1;
	temp2i=P1[i].temp2;
	temp3i=P1[i].temp3;
	temp_avg=(temp1i+temp2i+temp3i+tempi)/4;
	ptypei=P1[i].p_type;

	pri=poisson_ratio(ptypei);
	Gi=shear_modulus(ptypei);
	ymi=Gi*(2*(1+pri));


	mi=P1[i].m;
	rii=P1[i].ri;
	rhoi=P1[i].rho;

	demidxi=P1[i].dem_idx;



	if(k_con_solve){
		

		kc0_i=conductivity(tempi,ptypei);
		alph0_i=kc0_i/(rhoi*heat_capacity(tempi,ptypei));
		diff_kc0_i=diff_conductivity_1(tempi,ptypei,0);

		kc1_i=conductivity(temp1i,ptypei);
		alph1_i=kc1_i/(rhoi*heat_capacity(tempi,ptypei));
		diff_kc1_i=diff_conductivity_1(tempi,ptypei,0);

		kc2_i=conductivity(temp2i,ptypei);
		alph2_i=kc2_i/(rhoi*heat_capacity(tempi,ptypei));
		diff_kc2_i=diff_conductivity_1(tempi,ptypei,0);

		kc3_i=conductivity(temp3i,ptypei);
		alph3_i=kc3_i/(rhoi*heat_capacity(tempi,ptypei));
		diff_kc3_i=diff_conductivity_1(tempi,ptypei,0);

		kci=conductivity(tempi,ptypei);
		cpi=heat_capacity(tempi,ptypei);
		alph_i=(kci)/(cpi*rhoi);
		emi=emissivity(tempi,ptypei);

		CTE=Effective_CTE(temp_avg,ptypei);


	}

	//search_range=k_search_kappa*hi;	// search range

	mri=(mi/rhoi);

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;


	//tmpx=tmpy=tmpz=tmptqx=tmptqy=tmptqz=0.0;
	//tmpdt=0.0;
	//tmpn=0.0;
	//tmpd=1.0; -ORIGINAL

	tmpx=tmpy=tmpz=tmptqx=tmptqy=tmptqz=0.0;
	tmpdt=0.0;
	tmpdt1=0.0;
	tmpdt2=0.0;
	tmpdt3=0.0;
	tmpdt_rad=0.0;
	tmpdt_con=0.0;
	tmpn=0.0;
	tmpd=1.0;

	Real qv;
	//qv=P1[i].vol_power;
	qv=0.0;


	Real dr1, dr2;
	int_t n_dr=3;

	dr1=(5.0/6.0)*radi/n_dr;
	dr2=(1.0/6.0)*radi;
	
	Real vol_matrix;
	vol_matrix=(4.0/3.0)*3.141592*(radi*radi*radi-(radi-dr2)*(radi-dr2)*(radi-dr2));
	

	Real A, B,temp_star;

	A=(2*kc1_i/dr1)/(2*kc1_i/dr1+kc0_i/dr2);
	B=1-A;
	temp_star=A*temp1i+B*tempi;

	tmpdt=diff_kc0_i/(kc0_i/alph0_i)*((tempi-temp_star)/(2*dr2))*((tempi-temp_star)/(2*dr2))+2*(alph0_i/(radi))*(tempi-temp_star)/(2*dr2)+alph0_i*(tempi-2*tempi+temp_star)/(dr2*dr2);
	tmpdt1=diff_kc1_i/(kc1_i/alph1_i)*((2*temp_star-temp1i-temp2i)/(2*dr1))*((2*temp_star-temp1i-temp2i)/(2*dr1))+(2*alph1_i/(2.5*dr1))*(2*temp_star-temp1i-temp2i)/(2*dr1)+alph1_i*(2*temp_star-3*temp1i+temp2i)/(dr1*dr1);
	tmpdt2=diff_kc2_i/(kc2_i/alph2_i)*((temp1i-temp3i)/(2*dr1))*((temp1i-temp3i)/(2*dr1))+2*(alph2_i/(1.5*dr1))*(temp1i-temp3i)/(2*dr1)+alph2_i*(temp1i-2*temp2i+temp3i)/(dr1*dr1);
	tmpdt3=diff_kc3_i/(kc3_i/alph3_i)*((temp2i-temp3i)/(2*dr1))*((temp2i-temp3i)/(2*dr1))+2*(alph3_i/(0.5*dr1))*((temp2i-temp3i)/(2*dr1))+alph3_i*(temp2i-temp3i)/(dr1*dr1);

	// P1[i].test3=tmpdt;
	//P1[i].test5=qv/(kc1_i/alph1_i);
	// P1[i].test4=temp_star;
	// P1[i].test6=tmpdt1;

	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;

				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						

						if (P1[j].p_type>1000){
							Real xj,yj,zj,radj, tdist,ovlp; //rr
							int_t ptypej, itypej;
							Real tempj;
							Real radij;
						
						
							int_t demidxj;
							
	
						
							xj=P1[j].x;
							yj=P1[j].y;
							zj=P1[j].z;
							radj=P1[j].rad;
							tempj=P1[j].temp;
							radij=1/(1/radi+1/radj);
	
							demidxj=P1[j].dem_idx;

							ptypej=P1[j].p_type;
							itypej=P1[j].i_type;

							//if (itypej>2) break;
			
							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
							ovlp=radi+radj-tdist;


							if((tdist<3.5*radi)&&(tdist>0.0)){			// 2*sqrt(3)*rad -- should be modified -- assumption ybj

								if (k_con_solve){
									//radiation heat transfer
									Real dc_r, dc_ar;		// double-cone radius, double-cone area
									Real sb_const=5.6704e-8;

									Real emj=emissivity(tempj,ptypej);
									Real alphai=radi/tdist;
									Real alphaj=radj/tdist;

									Real di_max;			// maximum rhoi/distij (rhoi:hypotenuse of i side of double-cone)
									Real sin_i, sin_j, cos_i, cos_j;		// cosine of each side angle of double-cone
									Real tdist_cr, radi_cr, radj_cr;		// distance between cross sections (doublecone - DEM sphere) , radius of each cross section
									Real z_cr;								// value for calculating viewing factor F_a'b' between cross sections
									Real F_cr, F_ij;								// vewing factor F_a'b' between cross sections (doublecone-DEM sphere), viewing factor F_ab between two DEM particles

									Real ar_i, ar_j, ar_ratio;				// area of each sphere surface within double cone region, ratio(ar_ratio=ar_b/ar_a)
									Real emij_inv;
									Real Y_ij;

										//if(((radj-radi)/radi)<1.0e-3){

											Real porosity=0.401038;
											dc_r=0.56*(2*radij)*pow((1-porosity),-1/3);			//double-cone radius, 2*radij=radius...
											dc_ar=3.14159*dc_r*dc_r;

	

										//emij_inv=(1-emi)/emi+(1-em)

										//}

									di_max=0.5*(sqrt(1+4.0*dc_ar/(3.14159*tdist*tdist*(1-(alphaj-alphai)*(alphaj-alphai))))-(alphaj-alphai));
									cos_i=(1-(alphaj-alphai)*(alphaj-alphai))/(2*di_max)-(alphaj-alphai);
									cos_j=(1-(alphaj-alphai)*(alphaj-alphai))/(2*(di_max+(alphaj-alphai)))+(alphaj-alphai);
									sin_i=sqrt(1-cos_i*cos_i);
									sin_j=sqrt(1-cos_j*cos_j);

									radi_cr=radi*sin_i;
									radj_cr=radj*sin_j;
									tdist_cr=tdist-radi*cos_i-radj*cos_j;
									z_cr=1+(4*radj_cr*radj_cr+tdist_cr*tdist_cr)/(radi_cr*radi_cr);
									F_cr=0.5*(z_cr-sqrt(z_cr*z_cr-4.0*radj_cr*radj_cr/(radi_cr*radi_cr)));
									F_ij=(1+cos_i)*F_cr*0.5;

									ar_i=2.0*3.14159*radi*radi*(1-cos_i);
									ar_j=2.0*3.14159*radj*radj*(1-cos_j);
									ar_ratio=ar_j/ar_i;

									Y_ij=F_ij+(1-F_ij)*(ar_ratio/(1+ar_ratio));
									emij_inv=(1-emi)/(emi*ar_i)+(1-emj)/(emj*ar_j)+1/(Y_ij*ar_i);

									
									//tmpdt=tmpdt-sb_const*(tempi*tempi*tempi*tempi-tempj*tempj*tempj*tempj)/(emij_inv*mi*cpi);
									tmpdt_rad+=-sb_const*(tempi*tempi*tempi*tempi-tempj*tempj*tempj*tempj)/(emij_inv*(rhoi*vol_matrix)*cpi);

								}

								



							if(tdist<2.01*radi){

								
							if(tdist>0&&ovlp>0){
								
								Real uxj,uyj,uzj,wxj,wyj,wzj,mj,rhoj;
								Real urot_xi, urot_yi, urot_zi, urot_xj, urot_yj, urot_zj;
								Real ymj, prj;
								Real ymij, mij;
								
								Real Gj, Gij;
								Real rcxi, rcyi, rczi, rcxj, rcyj, rczj;		// r vector from center of particle i,j to contact point C
								
								Real fsx, fsy, fsz;
								Real fssx, fssy, fssz, fsfx, fsfy, fsfz;
								Real sgn_fssx, sgn_fssy, sgn_fssz;
								Real mag_fss, mag_fsf;
	
								Real nx, ny, nz, urel_x, urel_y, urel_z, un_mag, urel_mag;
								Real un_x, un_y, un_z, us_x, us_y, us_z;
								Real dsxj, dsyj, dszj;
	
								Real sx, sy, sz;	// tangential normal vector
	
								int_t ct_idx_j=-1;		// -1: no free contact slot (>16 simultaneous contacts) -> skip history write

	
								
	
								uxj=P1[j].ux;
								uyj=P1[j].uy;
								uzj=P1[j].uz;
	
								wxj=P1[j].wx;
								wyj=P1[j].wy;
								wzj=P1[j].wz;
	
								mj=P1[j].m;
								
								//Real tmpovlp=min(ovlp,radi*0.01);
								Real tmpovlp = ovlp;
								
								rhoj=P1[j].rho;

								if(k_con_solve){
									Real kcj, kc_ij;
									kcj=conductivity(tempj,ptypej);
									kc_ij=1/(1/kci+1/kcj);
									//tmpdt+=-4.0*kc_ij*sqrt(radij*ovlp)*(tempi-tempj);
									//tmpdt+=-4.0*kc_ij*sqrt(2*radij*ovlp-ovlp*ovlp)*(tempi-tempj)/(mi*cpi);

									tmpdt_con+=-4.0*kc_ij*sqrt((2*ovlp)*(radij-0.5*tmpovlp))*(tempi-tempj)/((rhoi*vol_matrix)*cpi);
									
								}
	
						

								prj=poisson_ratio(ptypej);
								Gj=shear_modulus(ptypej);
								ymj=Gj*(2*(1+prj));
	
								dsxj=dsyj=dszj=0.0;
	
	
	
								if(P1[i].ct_idx[0]==demidxj){
									dsxj=P1[i].del_s[0][0];
									dsyj=P1[i].del_s[0][1];
									dszj=P1[i].del_s[0][2];
									ct_idx_j=0;
								}
								else if(P1[i].ct_idx[1]==demidxj){
									dsxj=P1[i].del_s[1][0];
									dsyj=P1[i].del_s[1][1];
									dszj=P1[i].del_s[1][2];
									ct_idx_j=1;
								}
								else if(P1[i].ct_idx[2]==demidxj){
									dsxj=P1[i].del_s[2][0];
									dsyj=P1[i].del_s[2][1];
									dszj=P1[i].del_s[2][2];
									ct_idx_j=2;
								}
								else if(P1[i].ct_idx[3]==demidxj){
									dsxj=P1[i].del_s[3][0];
									dsyj=P1[i].del_s[3][1];
									dszj=P1[i].del_s[3][2];
									ct_idx_j=3;
								}
								else if(P1[i].ct_idx[4]==demidxj){
									dsxj=P1[i].del_s[4][0];
									dsyj=P1[i].del_s[4][1];
									dszj=P1[i].del_s[4][2];
									ct_idx_j=4;
								}
								else if(P1[i].ct_idx[5]==demidxj){
									dsxj=P1[i].del_s[5][0];
									dsyj=P1[i].del_s[5][1];
									dszj=P1[i].del_s[5][2];
									ct_idx_j=5;
								}
								else if(P1[i].ct_idx[6]==demidxj){
									dsxj=P1[i].del_s[6][0];
									dsyj=P1[i].del_s[6][1];
									dszj=P1[i].del_s[6][2];
									ct_idx_j=6;
								}
								else if(P1[i].ct_idx[7]==demidxj){
									dsxj=P1[i].del_s[7][0];
									dsyj=P1[i].del_s[7][1];
									dszj=P1[i].del_s[7][2];
									ct_idx_j=7;
								}
								else if(P1[i].ct_idx[8]==demidxj){
									dsxj=P1[i].del_s[8][0];
									dsyj=P1[i].del_s[8][1];
									dszj=P1[i].del_s[8][2];
									ct_idx_j=8;
								}
								else if(P1[i].ct_idx[9]==demidxj){
									dsxj=P1[i].del_s[9][0];
									dsyj=P1[i].del_s[9][1];
									dszj=P1[i].del_s[9][2];
									ct_idx_j=9;
								}
								else if(P1[i].ct_idx[10]==demidxj){
									dsxj=P1[i].del_s[10][0];
									dsyj=P1[i].del_s[10][1];
									dszj=P1[i].del_s[10][2];
									ct_idx_j=10;
								}
								else if(P1[i].ct_idx[11]==demidxj){
									dsxj=P1[i].del_s[11][0];
									dsyj=P1[i].del_s[11][1];
									dszj=P1[i].del_s[11][2];
									ct_idx_j=11;
								}
								else if(P1[i].ct_idx[12]==demidxj){
									dsxj=P1[i].del_s[12][0];
									dsyj=P1[i].del_s[12][1];
									dszj=P1[i].del_s[12][2];
									ct_idx_j=12;
								}
								else if(P1[i].ct_idx[13]==demidxj){
									dsxj=P1[i].del_s[13][0];
									dsyj=P1[i].del_s[13][1];
									dszj=P1[i].del_s[13][2];
									ct_idx_j=13;
								}
								else if(P1[i].ct_idx[14]==demidxj){
									dsxj=P1[i].del_s[14][0];
									dsyj=P1[i].del_s[14][1];
									dszj=P1[i].del_s[14][2];
									ct_idx_j=14;
								}
								else if(P1[i].ct_idx[15]==demidxj){
									dsxj=P1[i].del_s[15][0];
									dsyj=P1[i].del_s[15][1];
									dszj=P1[i].del_s[15][2];
									ct_idx_j=15;
								}
								// else if(P1[i].ct_idx[16]==demidxj){
								// 	dsxj=P1[i].del_s[16][0];
								// 	dsyj=P1[i].del_s[16][1];
								// 	dszj=P1[i].del_s[16][2];
								// 	ct_idx_j=16;
								// }
								// else if(P1[i].ct_idx[17]==demidxj){
								// 	dsxj=P1[i].del_s[17][0];
								// 	dsyj=P1[i].del_s[17][1];
								// 	dszj=P1[i].del_s[17][2];
								// 	ct_idx_j=17;
								// }
								// else if(P1[i].ct_idx[18]==demidxj){
								// 	dsxj=P1[i].del_s[18][0];
								// 	dsyj=P1[i].del_s[18][1];
								// 	dszj=P1[i].del_s[18][2];
								// 	ct_idx_j=18;
								// }
								// else if(P1[i].ct_idx[19]==demidxj){
								// 	dsxj=P1[i].del_s[19][0];
								// 	dsyj=P1[i].del_s[19][1];
								// 	dszj=P1[i].del_s[19][2];
								// 	ct_idx_j=19;
								// }
								else{
		
									if(P1[i].ct_idx[0]<1){
										TP1[i].ct_idx[0]=demidxj;
										ct_idx_j=0;
									}
									else if(P1[i].ct_idx[1]<1){
										TP1[i].ct_idx[1]=demidxj;
										ct_idx_j=1;
									}
									else if(P1[i].ct_idx[2]<1){
										TP1[i].ct_idx[2]=demidxj;
										ct_idx_j=2;
									}
									else if(P1[i].ct_idx[3]<1){
										TP1[i].ct_idx[3]=demidxj;
										ct_idx_j=3;
									}
									else if(P1[i].ct_idx[4]<1){
										TP1[i].ct_idx[4]=demidxj;
										ct_idx_j=4;
									}
									else if(P1[i].ct_idx[5]<1){
										TP1[i].ct_idx[5]=demidxj;
										ct_idx_j=5;
									}
									else if(P1[i].ct_idx[6]<1){
										TP1[i].ct_idx[6]=demidxj;
										ct_idx_j=6;
									}
									else if(P1[i].ct_idx[7]<1){
										TP1[i].ct_idx[7]=demidxj;
										ct_idx_j=7;
									}
									else if(P1[i].ct_idx[8]<1){
										TP1[i].ct_idx[8]=demidxj;
										ct_idx_j=8;
									}
									else if(P1[i].ct_idx[9]<1){
										TP1[i].ct_idx[9]=demidxj;
										ct_idx_j=9;
									}
									else if(P1[i].ct_idx[10]<1){
										TP1[i].ct_idx[10]=demidxj;
										ct_idx_j=10;
									}
									else if(P1[i].ct_idx[11]<1){
										TP1[i].ct_idx[11]=demidxj;
										ct_idx_j=11;
									}
									else if(P1[i].ct_idx[12]<1){
										TP1[i].ct_idx[12]=demidxj;
										ct_idx_j=12;
									}
									else if(P1[i].ct_idx[13]<1){
										TP1[i].ct_idx[13]=demidxj;
										ct_idx_j=13;
									}
									else if(P1[i].ct_idx[14]<1){
										TP1[i].ct_idx[14]=demidxj;
										ct_idx_j=14;
									}
									else if(P1[i].ct_idx[15]<1){
										TP1[i].ct_idx[15]=demidxj;
										ct_idx_j=15;
									}
									// else if(P1[i].ct_idx[16]<1){
									// 	TP1[i].ct_idx[16]=demidxj;
									// 	ct_idx_j=16;
									// }
									// else if(P1[i].ct_idx[17]<1){
									// 	TP1[i].ct_idx[17]=demidxj;
									// 	ct_idx_j=17;
									// }
									// else if(P1[i].ct_idx[18]<1){
									// 	TP1[i].ct_idx[18]=demidxj;
									// 	ct_idx_j=18;
									// }
									// else if(P1[i].ct_idx[19]<1){
									// 	TP1[i].ct_idx[19]=demidxj;
									// 	ct_idx_j=19;
									// }
		
		
								}
						
								
	
								nx = (xj-xi)/tdist;		// normal vector in normal direction
								ny = (yj-yi)/tdist;
								nz = (zj-zi)/tdist;
	
								rcxi = (radi-0.5*tmpovlp)*nx;		// r vector (from center of particle i to contact point C)
								rcyi = (radi-0.5*tmpovlp)*ny;	
								rczi = (radi-0.5*tmpovlp)*nz;	
								
	
								rcxj = -(radj-0.5*tmpovlp)*nx;		// r vector (from center of particle j to contact point C)
								rcyj = -(radj-0.5*tmpovlp)*ny;
								rczj = -(radj-0.5*tmpovlp)*nz;
	
								urot_xi=wyi*rczi-wzi*rcyi;
								urot_yi=wzi*rcxi-wxi*rczi;
								urot_zi=wxi*rcyi-wyi*rcxi;
	
								urot_xj=wyj*rczj-wzj*rcyj;
								urot_yj=wzj*rcxj-wxj*rczj;
								urot_zj=wxj*rcyj-wyj*rcxj;
	
								urel_x = (uxj+urot_xj)-(uxi+urot_xi);		// relative velocity in x direction
								urel_y = (uyj+urot_yj)-(uyi+urot_yi);		// relative velocity in y direction
								urel_z = (uzj+urot_zj)-(uzi+urot_zi);		// relative velocity in z direction

								urel_mag=sqrt(urel_x*urel_x+urel_y*urel_y+urel_z*urel_z);
	
								un_mag = nx*urel_x + ny*urel_y + nz*urel_z;		// magnitude of normal-direction relative velocity
	
								un_x = un_mag*nx;							// relative velocity in normal direction
								un_y = un_mag*ny;
								un_z = un_mag*nz;
	
								us_x = urel_x - un_x;						// relative velocity in tangential(shear) direction
								us_y = urel_y - un_y;
								us_z = urel_z - un_z;
	
								dsxj+= us_x*k_dt;
								dsyj+= us_y*k_dt;
								dszj+= us_z*k_dt;
	
								sx=us_x/(sqrt(us_x*us_x + us_y*us_y + us_z*us_z)+1e-20);	// normal vector in shear direction
								sy=us_y/(sqrt(us_x*us_x + us_y*us_y + us_z*us_z)+1e-20);
								sz=us_z/(sqrt(us_x*us_x + us_y*us_y + us_z*us_z)+1e-20);
	
	
								ymij=1/( ((1-pri*pri)/ymi) + ((1-prj*prj)/ymj) );
								Gij=1/( ((2-pri)/Gi) + ((2-prj)/Gj) );
								radij=1/(1/radi+1/radj);
								mij=1/(1/mi+1/mj);
	
								Real C_spr = (1.33333)*ymij*sqrt(radij)*sqrt(tmpovlp)*tmpovlp/mi;
								Real C_spr_s = 8.0*Gij*sqrt(radij*tmpovlp)/mi;
	
								Real Kn_dmp = 2*ymij*sqrt(radij*tmpovlp);
	
								Real K_dmp = sqrt(mij*Kn_dmp);
								Real C_dmp = sqrt(3.333333333)*0.0535184*K_dmp/mi;			//e=0.9 value 
								//Real C_dmp = sqrt(3.333333333)*0.09119*K_dmp/mi;				//e=0.75
	
								Real K_dmp_s = sqrt(mij*C_spr_s*mi);
								Real C_dmp_s = sqrt(3.333333333)*0.0535184*K_dmp_s/mi;		//e=0.9 ...
								//Real C_dmp_s = sqrt(3.333333333)*0.09119*K_dmp_s/mi;			//e=0.75
								
								// // Linear model
								// C_spr = 1000;
								// C_spr_s = 0.870056*C_spr;
								// C_dmp = sqrt(3.333333333)*0.0535184*sqrt(mij*C_spr*mi)/mi;
								// C_dmp_s = sqrt(3.333333333)*0.0535184*sqrt(mij*C_spr_s*mi)/mi;
							


								

									tmpx+= C_spr*(xi-xj)/tdist;		// normal elastic force (Hert's theory)
									tmpy+= C_spr*(yi-yj)/tdist;
									tmpz+= C_spr*(zi-zj)/tdist;

									tmpx+= C_dmp*un_x;				// normal damping term
									tmpy+= C_dmp*un_y;
									tmpz+= C_dmp*un_z;


								//if((itypej>1.5){
								//}
								//else{

								// 	tmpx+= C_spr*(xi-xj)/tdist/10.0;		// normal elastic force (Hert's theory)
								// 	tmpy+= C_spr*(yi-yj)/tdist/10.0;
								// 	tmpz+= C_spr*(zi-zj)/tdist/10.0;

								// 	tmpx+= C_dmp*un_x/10.0;				// normal damping term
								// 	tmpy+= C_dmp*un_y/10.0;
								// 	tmpz+= C_dmp*un_z/10.0;


								// }
	


								//if(itypej<2){

								//if(urel_mag>1.0e-2){
									
								fssx= C_spr_s*dsxj+C_dmp_s*us_x;				// tangential elastic force
								fssy= C_spr_s*dsyj+C_dmp_s*us_y;
								fssz= C_spr_s*dszj+C_dmp_s*us_z;
	
								//fssx= C_spr_s*dsxj;				// tangential elastic force
								//fssy= C_spr_s*dsyj;
								//fssz= C_spr_s*dszj;
	
								sgn_fssx=1.0*(fssx>=0)-1.0*(fssx<0);
								sgn_fssy=1.0*(fssy>=0)-1.0*(fssy<0);
								sgn_fssz=1.0*(fssz>=0)-1.0*(fssz<0);
	
								mag_fss=sqrt(fssx*fssx+fssy*fssy+fssz*fssz);
	
	
								fsfx=0.5*C_spr*sx;				// tangential force (should be modifed 2019.10.25)
								fsfy=0.5*C_spr*sy;
								fsfz=0.5*C_spr*sz;
	
								mag_fsf=sqrt(fsfx*fsfx+fsfy*fsfy+fsfz*fsfz);
	
								fsx=fssx*(mag_fss<mag_fsf)+sgn_fssx*abs(fsfx)*(mag_fss>=mag_fsf);
								fsy=fssy*(mag_fss<mag_fsf)+sgn_fssy*abs(fsfy)*(mag_fss>=mag_fsf);
								fsz=fssz*(mag_fss<mag_fsf)+sgn_fssz*abs(fsfz)*(mag_fss>=mag_fsf);

								// fsx=fssx;
								// fsy=fssy;
								// fsz=fssz;					
								
								//if(itypej<1.5){
									tmpx+= fsx;		// tangential force
									tmpy+= fsy; 
									tmpz+= fsz; 
								//}
								//else{
								//	tmpx+= fsx/10.0;		// tangential force
								//	tmpy+= fsy/10.0; 
								//	tmpz+= fsz/10.0; 
								//}
	

								//}
	

								//}
	
	
	
	
								// torque calculation
								tmptqx+= mi*(rcyi*fsz-rczi*fsy);
								tmptqy+= mi*(rczi*fsx-rcxi*fsz);
								tmptqz+= mi*(rcxi*fsy-rcyi*fsx);

								if(ct_idx_j>=0){		// only write tangential-history if a valid slot was found (guards >16 contacts)
									TP1[i].ct_idx[ct_idx_j]=demidxj;

									TP1[i].del_s[ct_idx_j][0]=dsxj;
									TP1[i].del_s[ct_idx_j][1]=dsyj;
									TP1[i].del_s[ct_idx_j][2]=dszj;
								}
							}

							else{

								if(P1[i].ct_idx[0]==demidxj){
									TP1[i].del_s[0][0]=0.0;
									TP1[i].del_s[0][1]=0.0;
									TP1[i].del_s[0][2]=0.0;
									TP1[i].ct_idx[0]=0;
								}
								else if(P1[i].ct_idx[1]==demidxj){
									TP1[i].del_s[1][0]=0.0;
									TP1[i].del_s[1][1]=0.0;
									TP1[i].del_s[1][2]=0.0;
									TP1[i].ct_idx[1]=0;
								}
								else if(P1[i].ct_idx[2]==demidxj){
									TP1[i].del_s[2][0]=0.0;
									TP1[i].del_s[2][1]=0.0;
									TP1[i].del_s[2][2]=0.0;
									TP1[i].ct_idx[2]=0;
								}
								else if(P1[i].ct_idx[3]==demidxj){
									TP1[i].del_s[3][0]=0.0;
									TP1[i].del_s[3][1]=0.0;
									TP1[i].del_s[3][2]=0.0;
									TP1[i].ct_idx[3]=0;
								}
								else if(P1[i].ct_idx[4]==demidxj){
									TP1[i].del_s[4][0]=0.0;
									TP1[i].del_s[4][1]=0.0;
									TP1[i].del_s[4][2]=0.0;
									TP1[i].ct_idx[4]=0;
								}
								else if(P1[i].ct_idx[5]==demidxj){
									TP1[i].del_s[5][0]=0.0;
									TP1[i].del_s[5][1]=0.0;
									TP1[i].del_s[5][2]=0.0;
									TP1[i].ct_idx[5]=0;
								}
								else if(P1[i].ct_idx[6]==demidxj){
									TP1[i].del_s[6][0]=0.0;
									TP1[i].del_s[6][1]=0.0;
									TP1[i].del_s[6][2]=0.0;
									TP1[i].ct_idx[6]=0;
								}
								else if(P1[i].ct_idx[7]==demidxj){
									TP1[i].del_s[7][0]=0.0;
									TP1[i].del_s[7][1]=0.0;
									TP1[i].del_s[7][2]=0.0;
									TP1[i].ct_idx[7]=0;
								}
								else if(P1[i].ct_idx[8]==demidxj){
									TP1[i].del_s[8][0]=0.0;
									TP1[i].del_s[8][1]=0.0;
									TP1[i].del_s[8][2]=0.0;
									TP1[i].ct_idx[8]=0;
								}
								else if(P1[i].ct_idx[9]==demidxj){
									TP1[i].del_s[9][0]=0.0;
									TP1[i].del_s[9][1]=0.0;
									TP1[i].del_s[9][2]=0.0;
									TP1[i].ct_idx[9]=0;
								}
								else if(P1[i].ct_idx[10]==demidxj){
									TP1[i].del_s[10][0]=0.0;
									TP1[i].del_s[10][1]=0.0;
									TP1[i].del_s[10][2]=0.0;
									TP1[i].ct_idx[10]=0;
								}
								else if(P1[i].ct_idx[11]==demidxj){
									TP1[i].del_s[11][0]=0.0;
									TP1[i].del_s[11][1]=0.0;
									TP1[i].del_s[11][2]=0.0;
									TP1[i].ct_idx[11]=0;
								}
								else if(P1[i].ct_idx[12]==demidxj){
									TP1[i].del_s[12][0]=0.0;
									TP1[i].del_s[12][1]=0.0;
									TP1[i].del_s[12][2]=0.0;
									TP1[i].ct_idx[12]=0;
								}
								else if(P1[i].ct_idx[13]==demidxj){
									TP1[i].del_s[13][0]=0.0;
									TP1[i].del_s[13][1]=0.0;
									TP1[i].del_s[13][2]=0.0;
									TP1[i].ct_idx[13]=0;
								}
								else if(P1[i].ct_idx[14]==demidxj){
									TP1[i].del_s[14][0]=0.0;
									TP1[i].del_s[14][1]=0.0;
									TP1[i].del_s[14][2]=0.0;
									TP1[i].ct_idx[14]=0;
								}
								else if(P1[i].ct_idx[15]==demidxj){
									TP1[i].del_s[15][0]=0.0;
									TP1[i].del_s[15][1]=0.0;
									TP1[i].del_s[15][2]=0.0;
									TP1[i].ct_idx[15]=0;
								}
								// else if(P1[i].ct_idx[16]==demidxj){
								// 	TP1[i].del_s[16][0]=0.0;
								// 	TP1[i].del_s[16][1]=0.0;
								// 	TP1[i].del_s[16][2]=0.0;
								// 	TP1[i].ct_idx[16]=0;
								// }
								// else if(P1[i].ct_idx[17]==demidxj){
								// 	TP1[i].del_s[17][0]=0.0;
								// 	TP1[i].del_s[17][1]=0.0;
								// 	TP1[i].del_s[17][2]=0.0;
								// 	TP1[i].ct_idx[17]=0;
								// }
								// else if(P1[i].ct_idx[18]==demidxj){
								// 	TP1[i].del_s[18][0]=0.0;
								// 	TP1[i].del_s[18][1]=0.0;
								// 	TP1[i].del_s[18][2]=0.0;
								// 	TP1[i].ct_idx[18]=0;
								// }
								// else if(P1[i].ct_idx[19]==demidxj){
								// 	TP1[i].del_s[19][0]=0.0;
								// 	TP1[i].del_s[19][1]=0.0;
								// 	TP1[i].del_s[19][2]=0.0;
								// 	TP1[i].ct_idx[19]=0;
								// }
	
	
							}
	

							}

						}
	
						}


					}			
				}
			}
		}
	}
	
	
	// y-directional gravitational force
	if(k_fg_solve){

		tmpz+=-9.8;	
		//tmpz=-9.8;

	} 

	//if(k_boussinesq_solve) tmpy+=Gravitational_CONST*(alpha_T*(tempi-T_ref0));

	if(k_con_solve){

		P1[i].test1=tmpdt_rad;
		P1[i].test2=tmpdt_con;
		tmpdt=tmpdt+tmpdt_rad+tmpdt_con;

		P3[i].dtemp=tmpdt;					// 나머지 입자들
		P3[i].dtemp1=tmpdt1;
		P3[i].dtemp2=tmpdt2;
		P3[i].dtemp3=tmpdt3;

		//P3[i].drad=radi*CTE*(tmpdt+tmpdt1+tmpdt2+tmpdt3)/4;
		P3[i].drad=0.0;
	}


	// if abs(tmpx)>1.0e5 tmpx=tmpx*1.0e5/abs(tmpx);
	// if abs(tmpy)>1.0e5 tmpy=tmpy*1.0e5/abs(tmpy);
	// if abs(tmpz)>1.0e5 tmpz=tmpz*1.0e5/abs(tmpz);


	// contact force calculation
	P3[i].ftotalx=tmpx;
	P3[i].ftotaly=tmpy;
	P3[i].ftotalz=tmpz;


	// assume packed bed - zero force 
	// P3[i].ftotalx=0.0;
	// P3[i].ftotaly=0.0;
	// P3[i].ftotalz=0.0;

	// // torque calculation
	P3[i].torqx = tmptqx/rii;
	P3[i].torqy = tmptqy/rii;
	P3[i].torqz = tmptqz/rii;

	// torque calculation
	// P3[i].torqx = 0.0;
	// P3[i].torqy = 0.0;
	// P3[i].torqz = 0.0;

}	