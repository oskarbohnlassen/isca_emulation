import xarray
import numpy as np
from xarray import ufuncs as xruf
from atmos import diffz
import area_average as aav
import pdb
import xarray as xar
from scipy.interpolate import CubicSpline

def pot_temp(thd_data, model_params, use_virt_temp=False):

    pref=model_params['pref']
    kappa=model_params['kappa']
    
    if use_virt_temp:
        try:
            temp=thd_data['virt_temp']
        except KeyError:
            virt_temp(thd_data, model_params)
            temp=thd_data['virt_temp']
        name_out = 'virt_ptemp'
    else:
        temp=thd_data['temp']
        name_out = 'ptemp'
    
    pot_temp=temp*np.power((pref/thd_data.pfull),kappa)

    thd_data[name_out]=(('time','pfull','lat','lon'),pot_temp)

def pot_temp_individual( temp, pfull, kappa = 2./7., pref=1000.):

    theta = temp * np.power((pref/pfull),kappa)

    return theta

def brunt_vas_freq(dataset, model_params, use_virt_temp=False):

    grav = model_params['g']

    if use_virt_temp:
        try:
            theta=dataset.virt_ptemp
        except AttributeError:
            pot_temp(dataset, model_params, use_virt_temp=True)
            theta=dataset.virt_ptemp    
        name_out = 'virt_nsqd'
    else:
        try:
            theta=dataset.ptemp
        except AttributeError:
            pot_temp(dataset, model_params)
            theta=dataset.ptemp
        name_out = 'nsqd'

    #Find relevant axes:
    pfull_loc = [x=='pfull' for x in theta.dims]

    pfull_idx = np.where(pfull_loc)[0][0]

    d_theta_dp = diffz(theta.values, dataset.pfull.values*100., axis=pfull_idx)
    

    nsqd_prefactor =  (-grav**2./theta)*(dataset.pfull * 100. / (model_params['rdgas']*dataset.temp))

    nsqd = nsqd_prefactor * d_theta_dp

    dataset[name_out]=(('time','pfull','lat','lon'),nsqd)

def eady_growth_rate_old(dataset, p_level_1, p_level_2, model_params):


    omega = model_params['omega']
    f_arr_temp = 2. * omega * xruf.sin(xruf.deg2rad(dataset.lat))

    times_arr, f_arr, lon_arr = np.meshgrid(dataset.time,f_arr_temp,dataset.lon,indexing='ij')



    nsqd=dataset.nsqd
    ucomp=dataset.ucomp
    height=dataset.height

    delta_z     =  height.sel(pfull=p_level_1,method='nearest') - height.sel(pfull=p_level_2,method='nearest')
    delta_u     =  ucomp.sel(pfull=p_level_1,method='nearest') - ucomp.sel(pfull=p_level_2,method='nearest')


    eady_array = (np.squeeze(0.31 * (xruf.fabs(f_arr) / xruf.sqrt(nsqd) )* xruf.fabs(delta_u / delta_z)))

#    unstable_nsqd_idx = (nsqd < 0.)
#    eady_array[unstable_nsqd_idx] = 0. #if nsqd profile is unstable, i.e. negative, then we zero out the eady growth rate in this region. 

    dataset['eady_gr']=(('time','lat','lon'),eady_array)
    
def eady_growth_rate_slightly_less_old(dataset, p_level_1, p_level_2, model_params):
    """Slightly modified after nsqd calc rewritten. Needs better vertical derivative."""

    omega = model_params['omega']
    f_arr_temp = 2. * omega * xruf.sin(xruf.deg2rad(dataset.lat))

    times_arr, f_arr, lon_arr = np.meshgrid(dataset.time,f_arr_temp,dataset.lon,indexing='ij')

    nsqd=dataset.nsqd.sel(pfull = 0.5*(p_level_1+p_level_2), method='nearest')
    ucomp=dataset.ucomp
    height=dataset.height

    delta_z     =  height.sel(pfull=p_level_1,method='nearest') - height.sel(pfull=p_level_2,method='nearest')
    delta_u     =  ucomp.sel(pfull=p_level_1,method='nearest') - ucomp.sel(pfull=p_level_2,method='nearest')


    eady_array = (np.squeeze(0.31 * (xruf.fabs(f_arr) / xruf.sqrt(nsqd) )* xruf.fabs(delta_u / delta_z)))

#    unstable_nsqd_idx = (nsqd < 0.)
#    eady_array[unstable_nsqd_idx] = 0. #if nsqd profile is unstable, i.e. negative, then we zero out the eady growth rate in this region. 

    dataset['eady_gr']=(('time','lat','lon'),eady_array)    

def eady_growth_rate(dataset, model_params):
    """Majorly modified after nsqd calc rewritten. Has better vertical derivative."""

    omega = model_params['omega']
    f_arr = 2. * omega * xruf.sin(xruf.deg2rad(dataset.lat))

    try:
        nsqd = dataset.virt_nsqd
    except:
        brunt_vas_freq(dataset, model_params, use_virt_temp=True)        

    ucomp=dataset.ucomp
    height=dataset.height

    ucomp.load()
    height.load()

    ucomp_tm = ucomp.mean('time')
    height_tm = height.mean('time')

    du_dz = xar.zeros_like(ucomp_tm)

    for lat_tick in range(dataset.lat.shape[0]):
        for lon_tick in range(dataset.lon.shape[0]):
            du_dz[:, lat_tick, lon_tick] = diffz(ucomp_tm[ :, lat_tick, lon_tick].values, height_tm[:, lat_tick, lon_tick].values)

    eady_array = 0.31 * (xruf.fabs(f_arr) / xruf.sqrt(nsqd.mean('time')) )* du_dz

    eady_array = eady_array.transpose('pfull','lat','lon')

    dataset['eady_gr']=(eady_array.dims,eady_array)    

def merid_sf(dataset, a=6371.0e3, g=9.8, start_integration_from_top = True, trap_rule=False, surf_pressure=None):
    """Calculate the mass streamfunction for the atmosphere.
    Based on a vertical integral of the meridional wind.
    Ref: Physics of Climate, Peixoto & Oort, 1992.  p158.
    `a` is the radius of the planet (default Earth 6371km).
    `g` is surface gravity (default Earth 9.8m/s^2).
    Returns an xarray DataArray of mass streamfunction.
    COPIED FROM https://github.com/ExeClim/ShareCode/blob/jpdev/execlim/analysis/mass_streamfunction.py on 16/05/16
    """
    vbar = dataset.vcomp.mean(('lon')).load()
    c = 2*np.pi*a*np.cos(vbar.lat*np.pi/180) / g
    # take a diff of half levels, and assign to pfull coordinates
    if trap_rule:

        c_all_time = np.zeros((len(dataset.time), len(dataset.lat)))

        for t_tick in range(len(dataset.time)):
            c_all_time[t_tick,:] = c

        merid_sf_trap_rule = np.zeros_like(vbar)

        p_values = dataset.pfull.values*100.

        if start_integration_from_top:
            delta_p = (-0. + p_values[0])
            mid_value = (0.+vbar[:,0,:])/2.
            merid_sf_trap_rule[:,0,:] = mid_value * delta_p *c_all_time

            for p_idx in range(1, len(p_values)):
                delta_p = (-p_values[p_idx-1] + p_values[p_idx])
                mid_value = (vbar[:,p_idx-1,:]+vbar[:,p_idx,:])/2.
                merid_sf_trap_rule[:,p_idx,:] = merid_sf_trap_rule[:,p_idx-1,:]+(mid_value*delta_p*c_all_time)
        else:

            delta_p = (surf_pressure - p_values[-1])
            mid_value = (0.+vbar[:,-1,:])/2.
            merid_sf_trap_rule[:,-1,:] = mid_value * delta_p *c_all_time

            for p_idx in range(len(p_values)-1, 0):
                delta_p = -(-p_values[p_idx-1] + p_values[p_idx])
                mid_value = (vbar[:,p_idx-1,:]+vbar[:,p_idx,:])/2.
                merid_sf_trap_rule[:,p_idx,:] = merid_sf_trap_rule[:,p_idx-1,:]+(mid_value*delta_p*c_all_time)

        dataset['merid_sf_trap_rule'] = (('time','pfull', 'lat'), merid_sf_trap_rule)

        merid_sf_sign = merid_sf_trap_rule / np.abs(merid_sf_trap_rule)

        dataset['log_merid_sf_trap_rule'] = (('time','pfull', 'lat'), merid_sf_sign*np.log(np.abs(merid_sf_trap_rule)))
        

    else:
        dp=xarray.DataArray(dataset.phalf.diff('phalf').values*100, coords=[('pfull', dataset.pfull)])
        if start_integration_from_top:
            product = vbar*dp
            product.load()
            merid_sf=c*np.cumsum(product, axis=product.dims.index('pfull'))
            merid_sf = merid_sf.transpose('time','pfull','lat')
            dataset['merid_sf']=(merid_sf.dims,merid_sf)        
        else:
            product = vbar[:,::-1,:]*dp[::-1]
            product.load()
            merid_sf=c*np.cumsum(product, axis=product.dims.index('pfull'))
            merid_sf = -1.*merid_sf[:,:,::-1]           
            merid_sf = merid_sf.transpose('time','pfull','lat')
            dataset['merid_sf_bot']=(merid_sf.dims,merid_sf) 
        
    
def lapse_rate(dataset, group_type='seasons'):

    temperature_bar_time_averaged = dataset['temp'].groupby(group_type).mean(('time','lon')).load()
    height_bar_time_averaged      = dataset['height'].groupby(group_type).mean(('time','lon')).load()
    
    dt_dz = np.zeros_like(temperature_bar_time_averaged.values)
    
    ntime, nplev, nlat = np.shape(temperature_bar_time_averaged)
        
    for time_tick in range(ntime):
        for lat_tick in range(nlat):
            t_profile = temperature_bar_time_averaged.values[time_tick,:,lat_tick]
            z_profile = height_bar_time_averaged.values[time_tick,:,lat_tick]
            
            t_prof = t_profile[np.newaxis,:]            
            dt_dz[time_tick,:,lat_tick] = diffz(t_prof, z_profile)
    
    dt_dz = dt_dz * -1000.
    
    dataset['dt_dz'] = ((group_type+'_ax','pfull','lat'), dt_dz)
    
def tropopause_height(dataset, model_params, group_type='seasons', do_high_res=False):

    tropopause_height_model_grid(dataset, model_params, group_type=group_type)
    if do_high_res:
        tropopause_height_high_res(dataset, model_params, group_type=group_type)        
        tropopause_press_high_res(dataset, model_params, group_type=group_type)        


def tropopause_height_model_grid(dataset, model_params, group_type='seasons'):

    try:
        dataset['dt_dz']
    except KeyError:
        print('lapse rate not present - calculating')
        lapse_rate(dataset, group_type)

    ntime, nplev, nlat = np.shape(dataset['dt_dz'])

    tropopause_height = np.zeros((ntime,nlat))        
        
    for time_tick in range(ntime):
        for lat_tick in range(nlat):

#             lapse_idx = nplev-1      
            try:
                middle_pfull = 0.5 * np.max(dataset.phalf.values)
            except:
                middle_pfull = 0.5 * np.max(dataset.pfull.values)

            lapse_idx = dataset.pfull.values.tolist().index(dataset.pfull.sel(pfull=middle_pfull,method='nearest'))
        
            lapse_val_temp = dataset['dt_dz'][time_tick,lapse_idx,lat_tick]
        
            if np.isfinite(lapse_val_temp):
                lapse_val = lapse_val_temp
            else:
                lapse_val = 10000.
                
            critical_value = 2.0 *model_params['g']/9.8

            while (lapse_val >= critical_value and lapse_idx>=0):
                lapse_val_temp = dataset['dt_dz'][time_tick,lapse_idx,lat_tick]
                if np.isfinite(lapse_val_temp):
                    lapse_val = lapse_val_temp
                else:
                    lapse_val = 10000.
                if lapse_val >=critical_value:          #Make sure that the index isn't incremented if the condition is already met 
                    lapse_idx = lapse_idx-1
            tropopause_height[time_tick,lat_tick] = dataset.pfull[lapse_idx]

    dataset['tropopause_height'] = ((group_type+'_ax','lat'), tropopause_height)


def tropopause_height_high_res(dataset, model_params, group_type='seasons'):

    try:
        dataset['dt_dz']
    except KeyError:
        print('lapse rate not present - calculating')
        lapse_rate(dataset, group_type)

    ntime, nplev, nlat = np.shape(dataset['dt_dz'])

    tropopause_height = np.zeros((ntime,nlat))        
        
    height_data = dataset['height'].mean('lon').load()
    lapse_rate_data = dataset['dt_dz'].load()

    z_levels = np.arange(0., height_data.max(), 100.)

    for time_tick in range(ntime):
        for lat_tick in range(nlat):

            cs_obj = CubicSpline(height_data[time_tick, ::-1, lat_tick], lapse_rate_data[time_tick, ::-1, lat_tick])
            high_res_dt_dz_profile = cs_obj(z_levels)

            lapse_idx = np.argmin(np.abs(z_levels-5000.*9.1/model_params['g']))
        
            lapse_val_temp = high_res_dt_dz_profile[lapse_idx]
        
            if np.isfinite(lapse_val_temp):
                lapse_val = lapse_val_temp
            else:
                lapse_val = 10000.
                
            critical_value = 2.0 *model_params['g']/9.8

            while (lapse_val >= critical_value and lapse_idx>=0):
                lapse_val_temp = high_res_dt_dz_profile[lapse_idx]
                if np.isfinite(lapse_val_temp):
                    lapse_val = lapse_val_temp
                else:
                    lapse_val = 10000.
                if lapse_val >=critical_value:          #Make sure that the index isn't incremented if the condition is already met 
                    lapse_idx = lapse_idx+1
            tropopause_height[time_tick,lat_tick] = z_levels[lapse_idx]

    dataset['tropopause_height_high_res'] = ((group_type+'_ax','lat'), tropopause_height)


def tropopause_press_high_res(dataset, model_params, group_type='seasons'):

    try:
        dataset['dt_dz']
    except KeyError:
        print('lapse rate not present - calculating')
        lapse_rate(dataset, group_type)

    ntime, nplev, nlat = np.shape(dataset['dt_dz'])

    tropopause_height = np.zeros((ntime,nlat))        
        
    pfull_data = dataset['pfull'].load()
    lapse_rate_data = dataset['dt_dz'].load()

    p_levels = np.arange(0., pfull_data.max(), 10.)

    for time_tick in range(ntime):
        for lat_tick in range(nlat):

            cs_obj = CubicSpline(pfull_data, lapse_rate_data[time_tick, :, lat_tick])
            high_res_dt_dz_profile = cs_obj(p_levels)

            try:
                middle_pfull = 0.5 * np.max(dataset.phalf.values)
            except:
                middle_pfull = 0.5 * np.max(dataset.pfull.values)

            lapse_idx = np.argmin(np.abs(p_levels-middle_pfull))

            lapse_val_temp = high_res_dt_dz_profile[lapse_idx]
        
            if np.isfinite(lapse_val_temp):
                lapse_val = lapse_val_temp
            else:
                lapse_val = 10000.
                
            critical_value = 2.0 *model_params['g']/9.8

            while (lapse_val >= critical_value and lapse_idx>=0):
                lapse_val_temp = high_res_dt_dz_profile[lapse_idx]
                if np.isfinite(lapse_val_temp):
                    lapse_val = lapse_val_temp
                else:
                    lapse_val = 10000.
                if lapse_val >=critical_value:          #Make sure that the index isn't incremented if the condition is already met 
                    lapse_idx = lapse_idx-1
            tropopause_height[time_tick,lat_tick] = p_levels[lapse_idx]

    dataset['tropopause_press_high_res'] = ((group_type+'_ax','lat'), tropopause_height)

def rh_offline(dataset, model_params):


    epsilon = model_params['rdgas']/ model_params['rvgas']
    one_minus_epsilon = 1.-epsilon
    
    es_array = saturation_vapour_press(dataset.temp.load(), model_params)
    
    rh_numerator = dataset['sphum']
    rh_denominator = (epsilon * es_array)/(dataset.pfull - one_minus_epsilon * es_array)
    
    rh_denom_denom = (dataset.pfull - one_minus_epsilon * es_array)
    
    #Nb don't need pfull in Pa as the constant we've used for e_0 is in hPa, so appropriate pressure units are hPa. Model will be using pfull in Pa and so es in Pa too.
    
    rh = 100.*rh_numerator / rh_denominator
        
    dataset['rh_offline'] = (dataset.sphum.dims, rh)
    
    dataset['rh_denom_denom'] = (dataset.sphum.dims, rh_denom_denom.transpose('time', 'pfull', 'lat', 'lon'))

    #Isca code has MAX(rh_denom_denom, esat), and I'm not sure that will ever do anything, as at low temperatures esat->0. and so rh_denom_denom -> pfull, and pfull > esat easily.

    dataset['esat'] = (dataset.sphum.dims, es_array)
    
    q = dataset['sphum'].load()
    
    partial_press_h20 = (q * dataset.pfull) / (q + epsilon*(1.-q))
    
    dataset['partial_pressure_h2o'] = (dataset.sphum.dims, partial_press_h20)

    rh_alt = 100. * partial_press_h20 / es_array
    dataset['rh_offline_alt'] = (dataset.sphum.dims, rh_alt)

    partial_press_h20_scaled = (q * (dataset.pfull*1000./dataset.pfull.max())) / (q + epsilon*(1.-q))
    
    rh_alt_scaled = 100. * partial_press_h20_scaled / es_array
    dataset['rh_offline_alt_scaled'] = (dataset.sphum.dims, rh_alt_scaled)

    # es_complex = saturation_vapour_press_complex(dataset.temp, model_params)

    # dataset['esat_complex'] = (dataset.sphum.dims, es_complex)

    # es_array_complex =dataset['esat_complex']/100. 

    # rh_denominator_complex = (epsilon * es_array_complex)/(dataset.pfull - one_minus_epsilon * es_array_complex)
        
    # #Nb don't need pfull in Pa as the constant we've used for e_0 is in hPa, so appropriate pressure units are hPa. Model will be using pfull in Pa and so es in Pa too.
    
    # rh_complex = 100.*rh_numerator / rh_denominator_complex
        
    # dataset['rh_offline_complex'] = (dataset.sphum.dims, rh_complex)
    

def saturation_vapour_press(temp, model_params, L_vapour = 2.44e6):

    e_0 = 6.12
    T_0 = 273.0
    
    es = e_0 * np.exp((L_vapour/model_params['rvgas']) * ( (1./T_0) - (1./temp)))
    
    return es

    
def saturation_vapour_press_complex(temp, model_params, L_vapour = 2.44e6):

    temp.load()
    es = xarray.zeros_like(temp)
    
    for t_tick in range(temp.shape[0]):
        for p_tick in range(temp.shape[1]):
            for lat_tick in range(temp.shape[2]):
                print(t_tick, p_tick, lat_tick)
                for lon_tick in range(temp.shape[3]):
                
                    t_val = temp[t_tick, p_tick, lat_tick, lon_tick]
                    es[t_tick, p_tick, lat_tick, lon_tick] = sat_vap_complex_calc(t_val)
    
    return es    

def sat_vap_complex_calc(t_in, TBASI=273., TBASW =373.):

    ESBASI = 610.71
    ESBASW = 101324.60
    

    if (t_in < TBASI):
        x = -9.09718*(TBASI/t_in-1.0) - 3.56654*np.log10(TBASI/t_in) \
         +0.876793*(1.0-t_in/TBASI) + np.log10(ESBASI)
        esice =10.**(x)
    else:
        esice = 0.

    # !  compute es over water greater than -20 c.
    # !  values over 100 c may not be valid
    # !  see smithsonian meteorological tables page 350.

    if (t_in > -20.+TBASI):
        x = -7.90298*(TBASW/t_in-1) + 5.02808*np.log10(TBASW/t_in) \
         -1.3816e-07*(10**((1-t_in/TBASW)*11.344)-1)        \
         +8.1328e-03*(10**((TBASW/t_in-1)*(-3.49149))-1)    \
         +np.log10(ESBASW)
        esh2o = 10.**(x)
    else:
        esh2o = 0.

    # !  derive blended es over ice and supercooled water between -20c and 0c

    if (t_in <= -20.+TBASI):
        es_out = esice
    elif (t_in >= TBASI):
        es_out = esh2o
    else:
        es_out = 0.05*((TBASI-t_in)*esice + (t_in-TBASI+20.)*esh2o)

    return es_out
    
def virt_temp(dataset,model_params):

    virt_fact = (model_params['rvgas']/model_params['rdgas']) - 1.0

    virt_temp = dataset['temp']*(1.+virt_fact*dataset['sphum'])
    
    dataset['virt_temp'] = (dataset.temp.dims, virt_temp)

def moist_static_energy(dataset, model_params):

    cp=model_params['cp_air']
    grav=model_params['g']    
    l_v = model_params['l_v']

    mse_temp   = cp*dataset['temp']
    mse_height = grav*dataset['height']
    mse_q      = l_v * dataset['sphum']

    mse =  mse_temp +  mse_height + mse_q

    mse_vars_dict = {'mse_temp':mse_temp, 'mse_height':mse_height, 'mse_q':mse_q, 'mse':mse}

    for mse_var_name in mse_vars_dict.keys():

        mse_var_data = mse_vars_dict[mse_var_name]

        mse_mean = mse_var_data.mean('lon')

        dataset[mse_var_name+'_3d'] = (mse_mean.dims, mse_mean)

        aav.vertical_integral(dataset, mse_var_name+'_3d', model_params, vertical_average=False)

    list_of_products_required=['vcomp_temp', 'vcomp_height', 'sphum_v']

    list_of_variables = [s for s in dataset.variables.keys()]

    are_products_in = [s in list_of_variables for s in list_of_products_required]

    if np.all(are_products_in):

        mse_temp_vcomp   = cp*dataset['vcomp_temp']
        mse_height_vcomp = grav*dataset['vcomp_height']
        mse_q_vcomp      = l_v * dataset['sphum_v']

        mse_vcomp_prod = mse_temp_vcomp + mse_height_vcomp + mse_q_vcomp

        mse_vars_dict = {'mse_temp_vcomp_prod':mse_temp_vcomp, 'mse_height_vcomp_prod':mse_height_vcomp, 'mse_q_vcomp_prod':mse_q_vcomp, 'mse_vcomp_prod':mse_vcomp_prod}

        for mse_var_name in mse_vars_dict.keys():

            mse_var_data = mse_vars_dict[mse_var_name]

            mse_mean = mse_var_data.mean('lon')

            dataset[mse_var_name+'_3d'] = (mse_mean.dims, mse_mean)

            aav.vertical_integral(dataset, mse_var_name+'_3d', model_params, vertical_average=False)

    mean_vcomp = dataset['vcomp'].mean('time').load()

    mse_temp_vcomp_mean   = cp*dataset['temp'].mean('time')*mean_vcomp
    mse_height_vcomp_mean = grav*dataset['height'].mean('time')*mean_vcomp
    mse_q_vcomp_mean      = l_v * dataset['sphum'].mean('time')*mean_vcomp

    mse_vcomp_mean = mse_temp_vcomp_mean + mse_height_vcomp_mean + mse_q_vcomp_mean

    mse_vars_dict = {'mse_temp_vcomp_mean':mse_temp_vcomp_mean, 'mse_height_vcomp_mean':mse_height_vcomp_mean, 'mse_q_vcomp_mean':mse_q_vcomp_mean, 'mse_vcomp_mean':mse_vcomp_mean}

    for mse_var_name in mse_vars_dict.keys():

        mse_var_data = mse_vars_dict[mse_var_name]

        mse_mean = mse_var_data.mean('lon')

        dataset[mse_var_name+'_3d'] = (mse_mean.dims, mse_mean)

        aav.vertical_integral(dataset, mse_var_name+'_3d', model_params, vertical_average=False)

def ssw(dataset, model_params):
    
    ucomp_timeseries = dataset['ucomp'].sel(pfull=10.,method='nearest').sel(lat=60., method='nearest').mean(('lon')).load()

    #Below code copied from atmos.py from ShareCode on 18th sept. Modified by sit.
    # uthr=0.0
    # u_below0 = np.where(ucomp_timeseries < uthr)[0]

    # if len(u_below0) > 0:
    #     idx = [u_below0[0]]
    #     for i in range(1, len(u_below0)):
    #         if u_below0[i] - idx[-1] > 20 and u_below0[i] - u_below0[i - 1] > 20:
    #             idx.append(u_below0[i])

    dataset['ssw_wind'] = (('time'), ucomp_timeseries)

    idx_below_previous = 0

    ssw_timing_arr=np.zeros(len(ucomp_timeseries))
    months_arr = dataset['months']
    idx = []
    time_of_negatives=[]

    for time_tick in range(0, len(ucomp_timeseries)):
        ucomp_val = ucomp_timeseries[time_tick]

        month_val = months_arr[time_tick]

        if month_val in [11,12,1,2,3]: #If it is November - March

            if ucomp_val < 0.0:

                if len(idx)==0:
                    idx = [time_tick]
                    ssw_timing_arr[time_tick] = 1

                is_it_20_days_since_last_ssw = (time_tick - idx[-1] > 20)

                if len(time_of_negatives)==0:
                    is_it_20_days_since_last_negative_wind= True
                else:
                    is_it_20_days_since_last_negative_wind = (time_tick - time_of_negatives[-1] > 20)

                if is_it_20_days_since_last_ssw and is_it_20_days_since_last_negative_wind :

                    is_final_warming = is_it_a_final_warming(time_tick, ucomp_timeseries, dataset)

                    if not is_final_warming:
                        ssw_timing_arr[time_tick] = 1
                        idx.append(time_tick)

                time_of_negatives.append(time_tick)

    dataset['ssw_timing'] = (('time'), ssw_timing_arr)


def generate_list_of_ssw_dates(dataset):

    ssw_timing_arr = dataset['ssw_timing']

    ssw_dates_list = []
    days_arr = dataset['day'].values
    months_arr = dataset['months'].values
    years_arr = dataset['years'].values

    month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

    for time_tick in range(len(ssw_timing_arr)):
        if ssw_timing_arr[time_tick]==1.0:

            day   = days_arr[time_tick]
            month = months_arr[time_tick]
            year  = years_arr[time_tick]

            date_str = str(day)+' ' +month_dict[month] + ' ' + str(year)
            ssw_dates_list.append(date_str)
    
    dataset.attrs['ssw_dates'] = ssw_dates_list

    ssw_freq_per_year = len(dataset.attrs['ssw_dates']) / len(np.unique(dataset.years))

    dataset.attrs['ssw_freq'] = ssw_freq_per_year

def is_it_a_final_warming(time_tick, ucomp_timeseries, dataset):

    days_arr = dataset['day'].values
    months_arr = dataset['months'].values
    years_arr = dataset['years'].values
    day_of_year_arr = dataset['dayofyear'].values

    current_day = days_arr[time_tick]
    current_month = months_arr[time_tick]
    current_year = years_arr[time_tick]    
    current_dayofyear = day_of_year_arr[time_tick]    

    #Time series between now and the end of April
        #If it's december or january then collect current days plus days from next year

    if current_month in [11,12]:
        for tick in range(len(ucomp_timeseries)):
            day = days_arr[tick]
            month = months_arr[tick]
            year = years_arr[tick]

            if day==30 and month==4 and year ==current_year+1:
                day_of_year_april_30th = day_of_year_arr[tick]

        try:
            day_of_year_april_30th
            end_of_dataset=False
        except:
            #If it hasn't managed to find the right april 30th, it is probably because we're at the end of the dataset, and 'next year' does not exist in the dataset. If so, assume it's not a final warming and carry on.
            end_of_dataset=True

        if end_of_dataset:
            is_it_a_final_warming_value = False #Highly unlikely to be final warming if current month is Nov or Dec
        else:
            this_year_values = ucomp_timeseries.where(dataset.years==current_year).where(dataset.dayofyear>=current_dayofyear)
            next_year_values = ucomp_timeseries.where(dataset.years==current_year+1).where(dataset.dayofyear<=day_of_year_april_30th)

            this_year_values_valid = this_year_values[np.where(np.isfinite(this_year_values))]
            next_year_values_valid = next_year_values[np.where(np.isfinite(next_year_values))]

            all_relevant_values_valid = np.append(this_year_values_valid, next_year_values_valid)

    elif current_month in [1,2,3]:

        for tick in range(len(ucomp_timeseries)):
            day = days_arr[tick]
            month = months_arr[tick]
            year = years_arr[tick]

            if day==30 and month==4 and year ==current_year:
                day_of_year_april_30th = day_of_year_arr[tick]

        all_relevant_values = ucomp_timeseries.where(dataset.years==current_year).where(dataset.dayofyear<=day_of_year_april_30th).where(dataset.dayofyear>=current_dayofyear)

        all_relevant_values_valid = all_relevant_values[np.where(np.isfinite(all_relevant_values))]
        end_of_dataset=False
    else:
        raise NotImplemented('Should not have reached this point')

    if not end_of_dataset:
        num_positive_days = np.where(all_relevant_values_valid>0.0)[0].shape[0]

        if num_positive_days > 10.:
            done=False
            num_consecutive_positives=0

            for i in range(len(all_relevant_values_valid)):
                if not done:
                    if all_relevant_values_valid[i] > 0.:
                        num_consecutive_positives = num_consecutive_positives+1
                        if num_consecutive_positives>=10:
                            is_it_a_final_warming_value=False
                            done=True
                    else:
                        done=False
                        num_consecutive_positives=0
                        is_it_a_final_warming_value=True

        else:
            is_it_a_final_warming_value = True

        #Are there more than 10 consecutive days of positive winds between now and then?
            #Identify days with positive winds
            #Are there 10 or more consecutive values?

        #If not then it's a final warming and so is not an SSW.

    return is_it_a_final_warming_value
