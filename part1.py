import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import geopy.distance

data2=pd.read_csv('/home/swat/Desktop/data2.csv')
data2.number=data2.number.replace(' None','0')

# Filtering out locations: Considering only Bangalore city 
bidadi=[12.810040,77.387221]
hoskote=[13.083670, 77.801490]
bommasandra=[12.789203, 77.713243]
hesargatta=[13.154739, 77.442054]

limit_lat=[12.789203,13.154739]
limit_lng=[77.387221,77.801490]

data2=data2[(data2.pick_lat >= limit_lat[0]) & (data2.pick_lat <= limit_lat[1]) & (data2.pick_lng >= limit_lng[0]) & (data2.pick_lng <= limit_lng[1])]
data2=data2.reset_index(drop=True)

#Adding 'distance covered' column
dist_col=[]
for m in range(data2.shape[0]):
    coord1=(data2.pick_lat[m],data2.pick_lng[m])
    coord2=(data2.drop_lat[m],data2.drop_lng[m])
    n=geopy.distance.geodesic(coord1, coord2).km
    dist_col.append(n)
data2['Distance']=dist_col

#Time Blocks
t_block=np.zeros(data2.shape[0])
t_block[data2[data2.Time<'07:00:00'].index.tolist()]=1
t_block[data2[(data2.Time<'11:00:00')&(data2.Time>'07:00:00')].index.tolist()]=2
t_block[data2[(data2.Time<'17:00:00')&(data2.Time>'11:00:00')].index.tolist()]=3
t_block[data2[(data2.Time<'21:00:00')&(data2.Time>'17:00:00')].index.tolist()]=4
t_block[data2[(data2.Time<'24:00:00')&(data2.Time>'21:00:00')].index.tolist()]=5

data2['T_block']=t_block

#Identify duplicates
data2['number'] = data2['number'].astype(float)
data2=data2.sort_values(['number','ts']).reset_index(drop=True)

t_block2=data2.T_block.tolist()

for i in range(1,len(data2.Time)):
    if data2.number[i]==data2.number[i-1]:
        if data2.Date[i]==data2.Date[i-1]:
            x=float(data2.Time[i].split(':')[1])
            y=float(data2.Time[i-1].split(':')[1])            
            if x < y+15:
                t_block2[i]=0

data2['T_block']=t_block2                
        
data2=data2.sort_values(['ts']).reset_index(drop=True)

#Remove duplicates
dup_df=data2[data2.T_block==0].reset_index(drop=True)
data2=data2[data2.T_block!=0].reset_index(drop=True)

# Location Cluster
pickup_clus=data2[['pick_lat','pick_lng']]
# Elbow method
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(pickup_clus)
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# K-means clustering with 8 clusters 
kmeans=KMeans(n_clusters=4,max_iter=300,random_state=12345)
kmeans.fit(pickup_clus)

centroids=kmeans.cluster_centers_
centroid_df=pd.DataFrame(centroids,columns=['Lat','Lng'])
 
      
#Labeling clusters
cluster_predict=[]

for j in range(data2.shape[0]):    
        loc=[(data2.pick_lat[j],data2.pick_lng[j])]
        p=kmeans.predict(loc)[0]
        cluster_predict.append(p)    

data2['Loc_Cluster']=cluster_predict

# Segregation 
data2['Count']=np.ones(data2.shape[0])

data_seg=data2.groupby(['Date','T_block','Loc_Cluster'])['Count','Distance'].sum().reset_index()

data_seg.to_csv(r'/home/swat/Desktop/seg.csv',header=True,index=False)
dup_df.to_csv(r'/home/swat/Desktop/duplicates.csv',header=True,index=False)
data2.to_csv(r'/home/swat/Desktop/no_dups.csv',header=True,index=False)






    
    
    