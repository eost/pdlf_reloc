'''
Define event clusters
'''

import os,h5py
import shutil as sh
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt


nevent  = 301
nsta_min = 5

# Decision matrix
CCB = np.zeros((nevent,nevent))
for station in ['BOR', 'BON', 'SNE', 'DSM', 'RVL', 'CSS', 'ENO', 'PHR', 'FJS', 'FLR', 'FOR', 'GPN', 'GPS', 'HIM', 'NSR', 'CRA', 'VIL', 'NTR', 'GBS', 'HDL,TTR', 'TKR', 'PRA', 'FRE', 'PBR', 'OBS', 'CIL', 'MAT', 'MAID', 'MVL', 'PRO', 'CAM', 'PJR', 'TXR', 'PER', 'BLE']:
    for channel in ['Z','N','E']:
        sta = station+channel
        f = open('cc_matrix.txt','rt')
        for l in f:
            items = l.strip().split()
            if items[2] != sta:
                continue
            i1 = int(items[0])
            i2 = int(items[1])
            cp = float(items[3])
            cn = float(items[4])
            if cp > 60.:
                CCB[i1,i2] += 1
                CCB[i2,i1] += 1
        f.close()

# Define clusters
eventid = np.arange(nevent)
clusters = []
for idx, ievent1 in enumerate(eventid):
    for ievent2 in eventid[idx+1:]:
        if CCB[ievent1,ievent2] > nsta_min:
            inacluster = False
            for i in range(len(clusters)):
                if ievent1 in clusters[i]:
                    if ievent2 not in clusters[i]:
                        clusters[i].append(ievent2)
                    inacluster = True
                elif ievent2 in clusters[i]:
                    if ievent1 not in clusters[i]:                    
                        clusters[i].append(ievent1)
                    inacluster = True
            if not inacluster:
                clusters.append([ievent1,ievent2])

# Merge clusters
cbk = clusters.copy()
mclusters = clusters.copy()
while 1:
    restart = False
    for i,cluster1 in enumerate(mclusters):    
        for ievent in cluster1:        
            for j,cluster2 in enumerate(mclusters):
                if i==j:
                    continue
                if ievent in cluster2:
                    mclusters[i].extend(cluster2)
                    mclusters[i] = list(np.unique(mclusters[i]))
                    print(j,len(mclusters))
                    mclusters.remove(mclusters[j])
                    restart = True
                    break
            if restart:
                break
        if restart:
            break
    if not restart:
        break
n = [len(cluster) for cluster in mclusters]

# Sort clusters
clusters = []
for i in np.argsort(n)[::-1]:
    clusters.append(mclusters[i])


print('%d clusters'%(len(clusters)))

# Get event location
fid=open('id_map_cc_matrix.txt','rt')
count = 0
lat = []
lon = []
dep = []
H   = []
for l in fid:
    items = l.strip().split()
    assert count == int(items[0])
    H.append(items[1])
    count += 1
    h = h5py.File(items[1], 'r')
    lat.append(h['HEADER'].attrs['LAT'])    
    lon.append(h['HEADER'].attrs['LONG'])
    dep.append(h['HEADER'].attrs['DEPH'])
    h.close()
fid.close()

lat = np.array(lat)
lon = np.array(lon)
dep = np.array(dep)

# Write clusters
if os.path.exists('clusters'):
    sh.rmtree('clusters')
os.mkdir('clusters')
for cid,cluster in enumerate(clusters):
    fid = open('clusters/c%d.txt'%(cid),'wt')
    for ievent in cluster:
        fid.write('%-4d %30s %9.4f %10.4f %9.4f\n'%(ievent,H[ievent],lat[ievent],lon[ievent],dep[ievent]))
    fid.close()
    
    


## Plot events
# Convert lon/lat to local UTM
string = '+proj=utm +lat_0={} +lon_0={} +ellps={}'.format(lat.mean(), lon.mean(), 'WGS84')
putm   = pp.Proj(string)
x,y = putm(lon,lat)
x /= 1000.
y /= 1000.

# Plot clusters
fig=plt.figure(figsize=[ 8.89,9.98])
for idx,cluster in enumerate(clusters):
    ax1 = plt.subplot(211)
    plt.plot(x[cluster],y[cluster],'o',label='c%d (%d events)'%(idx,len(cluster)))
    ax2 = plt.subplot(212)
    plt.plot(x[cluster],dep[cluster],'o')
ax1=plt.subplot(211)
plt.axis('equal')
plt.legend()
plt.title('All clusters (%d clusters)'%(len(clusters)))
plt.ylabel('North')
ax2=plt.subplot(212)
plt.axis('equal')
ax2.invert_yaxis()
plt.ylabel('Depth')
plt.xlabel('East')
plt.savefig('clusters/clusters.png')


# Plot clusters
plt.figure()
count = 0
for idx,cluster in enumerate(clusters):
    if len(cluster)>10:
        ax1 = plt.subplot(211)
        plt.plot(x[cluster],y[cluster],'o',label='c%d (%d events)'%(idx,len(cluster)))
        ax2 = plt.subplot(212)
        plt.plot(x[cluster],dep[cluster],'o')
        count += 1
ax1=plt.subplot(211)
plt.axis('equal')
plt.legend()
plt.title('Clusters with more than 10 events (%d clusters)'%(count))
plt.ylabel('North')
ax2=plt.subplot(212)
plt.axis('equal')
ax2.invert_yaxis()
plt.ylabel('Depth')
plt.xlabel('East')
    
# Plot decision matrix        
fig = plt.figure(figsize=[5.,4.5 ])
plt.pcolor(CCB,cmap=plt.get_cmap('viridis'))
plt.axis('equal')
#plt.clim(0,1)
plt.colorbar()
plt.xlabel('Event ID')
plt.ylabel('Event ID')

fig = plt.figure(figsize=[5.,4.5 ])
plt.pcolor(CCB,cmap=plt.get_cmap('viridis'))
plt.axis('equal')
plt.clim(0,5)
plt.colorbar()
plt.xlabel('Event ID')
plt.ylabel('Event ID')
plt.show()
