__host__ __device__ Real interp2(Real *x_data,Real *y_data,int size,Real x)
{

	Real y;
	int i;
	int end_idx=size-1;

	if(x_data[end_idx]<x){
		//y=y_data[end_idx]+(y_data[end_idx]-y_data[end_idx-1])/(x_data[end_idx]-x_data[end_idx-1])*(x-x_data[end_idx]);
		y=y_data[end_idx]-y_data[end_idx-1];
		y/=(x_data[end_idx]-x_data[end_idx-1]);
		y*=(x-x_data[end_idx]);
		y+=y_data[end_idx];
	}else if(x<=x_data[0]){
		//y=y_data[0]+(y_data[1]-y_data[0])/(x_data[1]-x_data[0])*(x-x_data[0]);
		y=y_data[1]-y_data[0];
		y/=(x_data[1]-x_data[0]);
		y*=(x-x_data[0]);
		y+=y_data[0];
	}else{
		for(i=0;i<size;i++){
			if((x_data[i]<x)&(x<=x_data[i+1])){
				//y=y_data[i]+(y_data[i+1]-y_data[i])/(x_data[i+1]-x_data[i])*(x-x_data[i]);
				y=y_data[i+1]-y_data[i];
				y/=(x_data[i+1]-x_data[i]);
				y*=(x-x_data[i]);
				y+=y_data[i];
				break;
			}
		}
	}
	return y;
}

__host__ __device__ Real conductivity(Real temp,uint_t p_type)
{
	Real cond;
	int index0,table_size0;
	p_type=abs(p_type);

	if (k_prop_table){
		index0=k_table_index[p_type];
		table_size0=k_table_size[p_type];
		cond=interp2(&k_Tab_T[index0],&k_Tab_k[index0],table_size0,temp);
	}
	else {
		if(p_type>1000){
			if(temp>=700.0){
				//cond=73.8428-0.0898607*temp+5.57553e-5*temp*temp-1.27420e-8*temp*temp*temp;		// conductivity of graphite at 700~1900 [K]
				//cond=2.64+0.989e-3*temp-3.65e-6*temp*temp+1.67e-9*temp*temp*temp;				// conductivity of ZrO2 at 700~1300 [K]
				cond=1.4; // Case 3-1-1 solid thermal conductivity [W/(m K)]
			}
			else{
				//cond=33.8899;			// conductivity of graphite at 700[K]
				//cond=2.11661;			// conductivity of ZrO2 at 700[K]
				cond=1.4; // Case 3-1-1 solid thermal conductivity [W/(m K)]
			}
		}
		else{												// 얘도 바꿔야할 것 같은데 아직 안 바꿈. 테이블 들어가서 바꿔야 할 듯? W/m/K = kg*m/k/s^3 = 1000 kg*mm/k/s^3
			cond=0.025; // Case 3-1-1 gas thermal conductivity [W/(m K)]
		}
	}
	return cond;
}

__host__ __device__ Real heat_capacity(Real temp,uint_t p_type)
{
	Real cp;
	p_type=abs(p_type);

	if(p_type>1000){
		if(p_type>1000){		// graphite

			cp=840;	// Case 3-1-1 solid specific heat [J/(kg K)]
			
		}
		else{
			cp=1068;
		}
		
	}
	else{
		cp=1010.0; // Case 3-1-1 gas specific heat [J/(kg K)]
	}
	//0.33;													// 얘도 바꿔야할 것 같은데 아직 안 바꿈. 테이블 들어가서 바꿔야 할 듯? W/m/K = kg*m/k/s^3 = 1000 kg*mm/k/s^3
	
	return cp;
}

__host__ __device__ Real viscosity(Real temp,uint_t p_type)
{
	Real vis;
	int index0,table_size0;
	p_type=abs(p_type);

	if (k_prop_table){
		index0=k_table_index[p_type];
		table_size0=k_table_size[p_type];
		vis=interp2(&k_Tab_T[index0],&k_Tab_vis[index0],table_size0,temp);
	}
	else{
		if (abs(p_type==3))	vis=VISCOSITY_AIR;
		else if (p_type==1) vis=VISCOSITY_AIR;	
		else vis=VISCOSITY_AIR;											// viscosity (Pa*s = kg/m/s = 0.001kg/mm/s) 테이블도 바꿔줘야 할 듯?
	}
	
	return vis;
}

__host__ __device__ Real thermal_expansion(Real temp,uint_t p_type)
{
	Real y;
	p_type=abs(p_type);

	if(p_type>1000){
		y=0.0;
		
	}
	else{
		y=0.0;
	}

	return y;
}

__host__ __device__ Real poisson_ratio(uint_t p_type)
{
	Real pr;
	//int index0,table_size0;
	p_type=abs(p_type);


	//if (p_type==DEM_SOLID)	pr=0.21;		// Al2O3 96% (Alumina Ceramic)
	if (p_type==DEM_SOLID)	pr=0.23;		// glass_bead
	else	pr=0.23;

	// if (p_type==DEM_SOLID)	pr=0.3;
	// else	pr=0.3;

	return pr;
}

__host__ __device__ Real shear_modulus(uint_t p_type)
{
	Real shm;
	//int index0,table_size0;
	p_type=abs(p_type);

	if (p_type==DEM_SOLID)	shm=8.3e8;			// graphite (Young's Modulus: 10 GPa, Poisson Ratio: 0.2)
	else	shm=8.3e8;

	// if (p_type==DEM_SOLID)	shm=3.84615e7;
	// else	shm=3.84615e7;

	return shm;
}

__host__ __device__ Real diff_conductivity_1(Real temp,uint_t p_type, uint_t material)
{
	Real diff_cond;
	int index0,table_size0;
	p_type=abs(p_type);

	if (k_prop_table){
		index0=k_table_index[p_type];
		table_size0=k_table_size[p_type];
		diff_cond=interp2(&k_Tab_T[index0],&k_Tab_k[index0],table_size0,temp);
	}
	else {
		if(p_type>1000){

			if(material==0){		// fuel

				//diff_cond=(-3*0.00000006*temp*temp-0.11+2*(135.6137/1000000)*temp);
				diff_cond=0;
			}

			if(material==1){		// graphite

				//diff_cond=(-3*0.000000066*temp*temp-0.1245+2*0.1517*0.001*temp);
				diff_cond=0;
			}
			}

		}
		return diff_cond;
}

__host__ __device__ Real emissivity(Real temp,uint_t p_type)
{
	Real em;
	p_type=abs(p_type);

	if(p_type>1000){
		if (temp>=700.0){
			em=0.505586+7.92943e-4*temp-5.228643e-7*temp*temp+1.10479e-10*temp*temp*temp;		// emissivity of graphite at 700~1900 [K]
			//em=-0.323+3.61e-3*temp-4.55e-6*temp*temp+1.67e-9*temp*temp*temp;					// emissivity of ZrO2 at 700~1300 [K]
		}
		else{
			em=0.842337;		// emissivity of graphite at 700[K]
			//em=0.54731;				// emissivity of ZrO2 at 700[K]
		}
		
	}
	else{
		//em=0.842337;
		em=0.54731;
	}
	//;													// 얘도 바꿔야할 것 같은데 아직 안 바꿈. 테이블 들어가서 바꿔야 할 듯? W/m/K = kg*m/k/s^3 = 1000 kg*mm/k/s^3
	
	return em;
}

__host__ __device__ Real Effective_CTE(Real temp,uint_t p_type)
{
	Real CTE_eff;
	int index0,table_size0;
	p_type=abs(p_type);

	if (k_prop_table){
		index0=k_table_index[p_type];
		table_size0=k_table_size[p_type];
		CTE_eff=interp2(&k_Tab_T[index0],&k_Tab_k[index0],table_size0,temp);
	}
	else {
		if(p_type>1000){

			CTE_eff=(0.00000000000003*temp*temp+0.00000000167*temp+0.000001826);
		}
		return CTE_eff;
	}	
}

__host__ __device__ Real conductivity_1(Real temp,uint_t p_type, uint_t material)
{
	Real cond;
	int index0,table_size0;
	p_type=abs(p_type);

	if (k_prop_table){
		index0=k_table_index[p_type];
		table_size0=k_table_size[p_type];
		cond=interp2(&k_Tab_T[index0],&k_Tab_k[index0],table_size0,temp);
	}
	else {
		if(p_type>1000){

			if(material==0){		// syllica (많은 것)

				//cond=-0.00000006*temp*temp*temp-0.11*temp+46.25+(135.6137/1000000)*temp*temp;
				cond=0.84;	// T=800[K]
			}

			if(material==1){		// Bronze (9개)

				//cond=-0.000000066*temp*temp*temp-0.1245*temp+51.317+0.1517*0.001*temp*temp;
				cond=55.0;	// T=800[K]
			}
			}

		}
		return cond;
}

__host__ __device__ Real diffusivity_1(Real temp,uint_t p_type, uint_t material)
{
	Real alph;
	int index0,table_size0;
	p_type=abs(p_type);

	if (k_prop_table){
		index0=k_table_index[p_type];
		table_size0=k_table_size[p_type];
		alph=interp2(&k_Tab_T[index0],&k_Tab_k[index0],table_size0,temp);
	}
	else {
		if(p_type>1000){

			if(material==0){		// syliica (많은 것)

				//alph=-0.00000000000004922*temp*temp*temp+0.000000000115*temp*temp-0.00000009663*temp+0.000036609;
				alph= 2.5000000000e-6;	// T=800[K]
			}

			if(material==1){		// Bronze (9개)

				//alph=-0.000000000000055*temp*temp*temp+0.0000000001276*temp*temp-0.00000010587*temp+0.000039318;
				alph= 1.7700000000000e-5;	// T=800[K]
			}
			}

		}
		return alph;
}