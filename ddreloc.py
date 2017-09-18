import numpy as np
import matplotlib.pyplot as plt # For plotting purposes
import os,scipy
import pyproj as pp
from scipy.sparse import linalg

# the intent of this program is to solve the DD relocation problem.
# This is a linear problem as d = Gm where d is the data vector containing the differential
# travel times, m is the parameter vector that contains the realtive postion of the pairs of events (in time and space) and G the Jacobian matrix it contains the ray parameters


# Need to check everywhere this stupid python convention where the first element in an array is 0 and not 1. # YOU ARE STUPID
# Make a structure containing 125 elements - with 3 fields containing 38 elements each

data_file = 'dt_cluster_ef_cc_c1.txt'
gmtxy_file = 'meca_az.gmtxy'
corr_th    = 0.4 # Correlation threshold

class event(object):
    '''
    Simple class describing azimuth, incidence angle and station names
    '''

    def __init__(self,index=None):
        '''
        Event initialization

        Parameters
        ----------        
        index : int (optional)
                index of the event (useful to build the G matrix)
        '''
        self.azm  = []
        self.toa  = []
        self.name = []
        self.idx = index
        
    def addsta(self,sta_name,azm,toa):
        '''
        Adding a station to the list
        Parameters
        ----------        
        name : str
               Station name
        azm : float
               Source to station azimuth
        toa : float
               Incidence angle (with respect to the vertical axis)
        '''
        self.name.append(sta_name)        
        self.azm.append(azm)
        self.toa.append(toa)

        


# Fill-out the event dictionary (replaces STA)
events = {}

# Read NLL outputs (be careful to change the azimuth azm = (180-azm) because we want from EQ to STA and not STA to EQ, note that this might be changed directly in NLL input - to be tested)
f=open(gmtxy_file,'r')

nevent = 0
ndata  = 0
event_idx_map = []
for line in f:

    # Split the line into columns
    items = line.strip().split()
    print(items,len(items))
    if len(items) != 6:
        print('Stop reading file at line %d: "%s"'%(ndata+1,line.strip()))
        break
        
    # Read event id
    evid = items[4]

    # Read Azm
    azimuth = float(items[0])
    azimuth = azimuth-180.
    if(azimuth<0):
        azimuth=azimuth+360
    ic = float(items[3])
    station = items[5]

    # Check if this event is already in the dictionary
    if evid not in events:
        events[evid] = event(index=nevent)
        event_idx_map.append(evid)
        nevent += 1

    # Fill-out the Event dictionary
    events[evid].addsta(station,azimuth,ic)
    ndata += 1

f.close()

# Set the P-wave speed
Vp=4000
# Set the S-wave speed
Vs=Vp/1.73

# Get the number of data points
dmax = sum(1 for line in open(data_file))

# Allocate data, G and covariance matrices
d = np.zeros((dmax,1))
sd_inv = np.zeros((dmax,1))
G = np.zeros((dmax,4*nevent))
Cm = np.eye(4*nevent)

# Open the time-delay data file
f=open(data_file,'r')
data_idx = 0
for line in f:
    # Read the line and get rid of the special characters 
    items = line.strip().split()

    # Read the station
    sta = items[4]

    # Read event IDs
    ev1 = items[2]
    ev2 = items[3]

    # Correlation coefficient
    R = float(items[5])/100.
    #if R < corr_th:
    #    print('Skipping station %s for event pair %s %s (Correlation coefficient = %.2f)'%(sta,ev1,ev2,R))
    #    continue
    
    # Read the delay (ms)
    dt =  float(items[7])/1000.

    # Read the uncertainty
    #s_dt = float(items[9])/1000.
    s_dt = (100-R*100.)/100 # An alternative?

    # check if ev1 and ev2 exists in the event dictionary
    if ev1 not in events:
        print('%s not in %s'%(ev1,gmtxy_file))
        continue
    if ev2 not in events:
        print('%s not in %s'%(ev2,gmtxy_file))
        continue

    # Find station in list
    found1=False
    for index1,sta1 in enumerate(events[ev1].name):
        if sta1 == sta[:-1]:
            found1 = True
            break
    found2 = False
    for index2,sta2 in enumerate(events[ev2].name):
        if sta2 == sta[:-1]:
            found2 = True
            break        

    # Fill out the G matrix
    if found1 and found2:
        azm1 = events[ev1].azm[index1]
        toa1 = events[ev1].toa[index1]

        azm2 = events[ev2].azm[index2]
        toa2 = events[ev2].toa[index2]

        # We consider the average (to be improved
        azm = (azm1 + azm2)/2.
        toa = (toa1 + toa2)/2.

        # Select the corresponding wave velocity
        if(sta[-1] == 'Z'): # P-wave velocity for horizontal channels
            c = Vp   
        else:               # S-wave velocity for vertical channels
            c = Vs

        # Compute the ray parameters
        A = np.sin(np.deg2rad(azm)) * np.sin(np.deg2rad(toa)) / c
        B = np.cos(np.deg2rad(azm)) * np.sin(np.deg2rad(toa)) / c
        C = np.cos(np.deg2rad(toa)) / c

        # Fill the data vector
        d[data_idx,0]  = dt

        # Fill the data uncertainties (inverse matrix)
        sd_inv[data_idx,0] = 1./s_dt
        #Cdinv[data_idx,data_idx] = 1./(np.power(s_dt,2))
        
        # Fill the Jacobian matrix
        G[data_idx,events[ev1].idx*4+0] = -A
        G[data_idx,events[ev1].idx*4+1] = -B
        G[data_idx,events[ev1].idx*4+2] = -C
        G[data_idx,events[ev1].idx*4+3] = -1.

        G[data_idx,events[ev2].idx*4+0] = A
        G[data_idx,events[ev2].idx*4+1] = B
        G[data_idx,events[ev2].idx*4+2] = C
        G[data_idx,events[ev2].idx*4+3] = 1.

        # Increment data index
        data_idx += 1
    else:
        print('%s not found in the event dictionary'%(sta))
# Close the file after we finish reading it
f.close()

# Now I can remove empty rows if they exist
I=np.all(G==0,axis=1)
G = G[~I,0:4*nevent]
d = d[~I,:]
nd = len(d)
#diag_cd = np.diag(Cdinv)
#Cdinv = np.diag(diag_cd[~I])
sd_inv = sd_inv[~I]
d2 = d*sd_inv
G2 = G*sd_inv

iid = np.linspace(0,nd,nd+1)
print(nd)

# We could also reduce the number fo columns for all events that are not used


# Read original location
f = open('clusters/c1.txt','rt')
ievents = []
idevents = []
lons    = []
lats    = []
zorg    = []
for l in f:
    items = l.strip().split()
    ievents.append(int(items[0]))
    idevents.append(os.path.basename(items[1])[:-5])
    lats.append(float(items[2]))
    lons.append(float(items[3]))
    zorg.append(float(items[4])*1000.)
    print(idevents[-1])
lats = np.array(lats)
lons = np.array(lons)
zorg = np.array(zorg)
string = '+proj=utm +lat_0={} +lon_0={} +ellps={}'.format(lats.mean(), lons.mean(), 'WGS84')
putm   = pp.Proj(string)
xorg,yorg = putm(lons,lats)
xorg -= xorg.mean()
yorg -= yorg.mean()
zorg -= zorg.mean()

print('MEAN AND STD (original location)')
print(xorg.mean(),xorg.std())
print(yorg.mean(),yorg.std())
print(zorg.mean(),zorg.std())

# Assigned a priori uncertainties on the parameters
# The parameter vector is identified as 
# m(0,1,2,3,4,5 ...,N) = (x_1,y_1,z_1,t0_1,x_2,...,t0_N);
# Because we are dealing with relative location we decided that all
# locations will be given relative to the first event of the group. In
# order to to so we assigned an extremely low a-priori uncertainty on the
# location of the first event.
# Actually here what is given is already CM^(-1) (the inverse of the
# diagonal matrix CM).
      
# Assign the a priori uncertainties (from the standard deviation of the original location)
sigma_m = np.array([xorg.std(),yorg.std(),zorg.std(),0.1]*nevent)
Cminv = np.diag(1./np.power(sigma_m,2)) # Both in meters and seconds (not very good)

# Set low uncertainties to all the parameters of the 1st event
Cminv[0,0]=1e9 
Cminv[1,1]=1e9
Cminv[2,2]=1e9
Cminv[3,3]=1e12

# Resolve the inverse problem
b = G2.T.dot(G2)+Cminv
B = G2.T.dot(d2)
x,resid,rank,s = np.linalg.lstsq(b,B)
res = np.abs(d -  G.dot(x))
res = res.flatten()
res_median = np.median(res)
#print(res_median*1000,np.mean(np.abs(res))*1000,nd)

# We will iterate and remove and each time step the data with the highest msifit
#screening = [5.,3.,1.5,0.9,0.8]
for iter in range(5):
    print('Iteration %d'%(iter))
    # Compute residuals
    res = np.abs(d -  G.dot(x))
    res = res.flatten()    
    # Compute the median residual 
    res_median = np.median(res)
    nres = res/(np.abs(d.flatten()))
    # Compute the weigthed misfit
    #wmsft = np.mean(res*sd_inv)
    print(res_median*1000,np.mean(np.abs(res))*1000,nd)
    # Find those data that do not well fit the model
    #I = np.where (nres < screening[iter])[0]
    I = np.where (res < 5*res_median)[0]
    # Remove bad data
    G = G[I,:]
    d = d[I,:]
    G2 = G2[I,:]
    d2 = d2[I,:]
    nd = len(d)    
    iid = np.asarray(iid[I])        
    b = G2.T.dot(G2)+Cminv
    B = G2.T.dot(d2)
    x,resid,rank,s = np.linalg.lstsq(b,B)
    
print(res_median*1000,np.mean(np.abs(res))*1000,nd)

# #
# res_list = []
# mod_list = []
# s_list   = np.exp(np.linspace(np.log(0.01),np.log(1.),20))
# for sigma in s_list:
#     print(sigma)
    
#     # Assign the a priori uncertainties
#     #Cminv = np.eye(4*nevent)
#     #Cminv = Cminv * (1./np.power(sigma,2)) # Both in meters and seconds (not very good)
#     sigma_m = np.array([xorg.std(),yorg.std(),zorg.std(),sigma]*nevent)
#     Cminv = np.diag(1./np.power(sigma_m,2)) # Both in meters and seconds (not very good)

#     # Set low uncertainties to all the parameters of the 1st event
#     Cminv[0,0]=1e9 
#     Cminv[1,1]=1e9
#     Cminv[2,2]=1e9
#     Cminv[3,3]=1e12

#     # Compute the first matrix 
#     b = G2.T.dot(G2)+Cminv
#     # Compute the second matrix
#     B = G2.T.dot(d2)
#     # Find the solution of the linear least square problem
#     x,resid,rank,s = np.linalg.lstsq(b,B)
    
#     # Compute residuals for each data
#     res = np.abs(d -  G.dot(x))
#     res = res.flatten()
    
#     # Compute the median residual 
#     res_median = np.median(res)    

#     #
#     res_list.append(res_median)

#     #
#     mod_list.append((x.T.dot(x))[0][0])


# plt.figure()
# plt.plot(res_list,mod_list,'-o')
# for i in range(s_list.size):
#     plt.text(res_list[i],mod_list[i],s_list[i])
# plt.show()

    
# At the last iteration
# Get the associated uncertainties and save them
s_m = np.linalg.inv(G2.T.dot(G2)+Cminv)
s_m = np.sqrt(np.diag(s_m))
s_m2=np.reshape(s_m,(nevent,4))
np.savetxt('test_s.out', s_m2, delimiter=' ')   

# Save the parameters vector 
x2=np.reshape(x,(nevent,4))
np.savetxt('test.out', x2, delimiter=' ') 
np.savetxt('test_res.out', res, delimiter=' ')
np.savetxt('test_d.out', iid.astype('int'),fmt='%d', delimiter=' ') 
plt.plot(d)
plt.plot(G.dot(x),'xr')

# Plot locations
plt.figure()

plt.subplot(211)
plt.plot(xorg,yorg,'x')
plt.plot(x2[:,0]-x2[:,0].mean(),x2[:,1]-x2[:,1].mean(),'o')
plt.axis('equal')

plt.subplot(212)
plt.plot(xorg,zorg,'x')
plt.plot(x2[:,0]-x2[:,0].mean(),x2[:,2]-x2[:,2].mean(),'o')
plt.axis('equal')
plt.show()



