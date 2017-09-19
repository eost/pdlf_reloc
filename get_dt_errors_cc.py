#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

id_map_file = 'id_map_cc_c1.txt'
dt_file     = 'dt_cluster_ef_cc_c1.txt'
o_dt_file   = 'dt_cluster_ef_cc_c1_std.txt'

# Read event ids
fid = open(id_map_file,'rt')
eventids = []
hdrs     = []
for l in fid:
    items = l.strip().split()
    eventids.append(int(items[0]))
    hdrs.append(items[1])
fid.close()

# Load delays
fid = open(dt_file,'rt')
dt_dict = {}
ev_pairs = []
for l in fid:
    # Parse data in line
    items = l.strip().split()
    R   = float(items[5])/100.
    tau = float(items[7])/1000.
    sta = items[4]
    ievent1 = int(items[0])
    ievent2 = int(items[1])
    # Append event pair to list
    if (ievent1,ievent2) not in ev_pairs:
        ev_pairs.append((ievent1,ievent2))
    # Complete the dt dictionary
    if sta not in dt_dict:
        dt_dict[sta] = {}
    dt_dict[sta][ievent1,ievent2] = [ tau,R]
    dt_dict[sta][ievent2,ievent1] = [-tau,R]
fid.close()

# Station list
stats = list(dt_dict.keys())

# Main loop over stations
for sta in stats:
    print(sta)
    # Loop over event pairs
    for ev_pair in ev_pairs:
        # Check if ev_pair is available for this station
        if ev_pair not in dt_dict[sta]:
            continue
        # Find triplets
        dt_sum = []
        for ievent in eventids:
            # Only take events that are not in ev_pair
            if ievent in ev_pair:
                continue
            # Pairs in the triplet
            t_pair1 = (ev_pair[1],ievent)
            t_pair2 = (ievent,ev_pair[0])
            if t_pair1 in dt_dict[sta] and t_pair2 in dt_dict[sta]:
                dt_sum.append(dt_dict[sta][ev_pair][0] + dt_dict[sta][t_pair1][0] + dt_dict[sta][t_pair2][0])
        if len(dt_sum) == 0:
            mean_dt = None
            med_dt  = None
            sl_dt   = None
        else:
            mean_dt = np.absolute(np.mean(dt_sum)) # Mean enclosure residual
            med_dt = np.absolute(np.median(dt_sum)) # Mean enclosure residual
            sl_dt = np.std(dt_sum)  # Standard deviation of enclosure residuals
            #if len(dt_sum)>55 and sta=='GPNN':
            #    bins = np.linspace(-0.02,0.02,40)
            #    plt.hist(dt_sum,bins=bins)
            #    plt.show()
        
        dt_dict[sta][ev_pair].append(mean_dt)
        dt_dict[sta][ev_pair].append(med_dt)        
        dt_dict[sta][ev_pair].append(sl_dt)
        dt_dict[sta][ev_pair].append(len(dt_sum))


# Output file
i_fid = open(dt_file,'rt')
o_fid = open(o_dt_file,'wt')
for l in i_fid:
    items = l.strip().split()
    sta = items[4]
    ievent1 = int(items[0])
    ievent2 = int(items[1])
    ev_pair = (ievent1,ievent2)
    if sta in dt_dict:
        if ev_pair in dt_dict[sta] and dt_dict[sta][ev_pair][-1] > 1:
            loop_error = dt_dict[sta][ev_pair][-2]/np.sqrt(float(dt_dict[sta][ev_pair][-1]))
            dt_error = np.sqrt(dt_dict[sta][ev_pair][-4]*dt_dict[sta][ev_pair][-4]+loop_error*loop_error)
            h1 = items[2]
            h2 = items[3]
            R1 = float(items[5])
            R2 = float(items[6])
            tau1 = float(items[7])
            tau2 = float(items[8])
            items.append(dt_error)
            s_out = '%4d %4d %20s %20s %7s %7.1f %7.1f %10.2f %10.2f %10.2f\n'%(ievent1,ievent2,h1,h2,sta,R1,R2,tau1,tau2,dt_error*1000.)
            o_fid.write(s_out)
i_fid.close()            
o_fid.close()

        
        
# Plot
for sta in stats:
    # Loop over event pairs
    l_dts_mean  = []
    l_dts_med   = []    
    ls_dts = []
    R_dts  = []
    for ev_pair in ev_pairs:
        if ev_pair in dt_dict[sta] and dt_dict[sta][ev_pair][-1] > 1:
            loop_error = dt_dict[sta][ev_pair][-2]/np.sqrt(float(dt_dict[sta][ev_pair][-1]))
            #if loop_error < 1.:
            l_dts_mean.append(dt_dict[sta][ev_pair][-4])
            l_dts_med.append(np.sqrt(dt_dict[sta][ev_pair][-4]*dt_dict[sta][ev_pair][-4]+loop_error*loop_error))
            ls_dts.append(loop_error)
            R_dts.append(dt_dict[sta][ev_pair][1])
            #print(dt_dict[sta][ev_pair][-1])
    print(sta,len(R_dts))
    plt.figure(figsize=[7.1125,12.3375])
    plt.subplot(211)
    plt.plot(R_dts,l_dts_mean,'o')
    #plt.errorbar(c_dts,l_dts,ls_dts,fmt='o')
    plt.title(sta)
    plt.grid('on')
    plt.xlabel('Correlation')
    plt.ylabel('Enclosure residual mean')
    plt.subplot(212)
    plt.plot(R_dts,l_dts_med,'o')
    plt.title(sta)
    plt.grid('on')
    plt.xlabel('Correlation')
    plt.ylabel('Enclosure residual median')    
    plt.show()
