import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

seg=pd.read_csv('/home/swat/Desktop/seg.csv')
dup_df=pd.read_csv('/home/swat/Desktop/duplicates.csv')
no_dups=pd.read_csv('/home/swat/Desktop/no_dups.csv')

g_date=seg.groupby(['Date'])['Count'].sum().reset_index()
sns.barplot(x='Date',y='Count',data=g_date,palette='hls')
plt.title('barchart of #rides requested')
plt.savefig('plot_1.png')
plt.show()


g_tblock=seg.groupby(['T_block'])['Count'].sum().reset_index()
sns.barplot(x='T_block',y='Count',data=g_tblock,palette='hls')
plt.title('No.of Rides requested')
plt.savefig('plot_2.png')
plt.show()


g_loc=seg.groupby(['Loc_Cluster'])['Count'].sum().reset_index()
sns.barplot(x='Loc_Cluster',y='Count',data=g_loc,palette='hls')
plt.title('No.of Rides requested')
plt.savefig('plot_3.png')
plt.show()

import datetime
import calendar 
for lab, row in seg.iterrows():
    f= datetime.datetime.strptime(row['Date'],'%Y-%m-%d').weekday()
    seg.loc[lab,'Day']=calendar.day_name[f]

g_day=seg.groupby(['Day'])['Count'].sum().reset_index()
sns.barplot(x='Day',y='Count',data=g_day,palette='hls')
plt.title('No.of Rides requested')
plt.savefig('plot_4.png')
plt.show()


g_dt=seg.groupby(['Day','T_block'])['Count'].sum().reset_index()
g_dt2= g_dt.pivot(index='Day', columns='T_block', values='Count')
g_dt2.plot(kind='bar')
#g_dt2.plot(kind='bar',stacked=True)
plt.title('No.of Rides requested')
plt.xlabel('Day')
plt.ylabel('Count')
plt.savefig('plot_5.png')
plt.show()

g_tl=seg.groupby(['T_block','Loc_Cluster'])['Count'].sum().reset_index()
g_tl2= g_tl.pivot(index='T_block', columns='Loc_Cluster', values='Count')
g_tl2.plot(kind='bar')
#g_dt2.plot(kind='bar',stacked=True)
plt.title('No.of Rides requested')
plt.xlabel('Time Block')
plt.ylabel('Count')
plt.savefig('plot_6.png')
plt.show()

g_id=no_dups.groupby(['number'])['Count'].sum().reset_index()
g_id=g_id.sort_values('Count',ascending=False).reset_index(drop=True)
sns.barplot(x='number',y='Count',data=g_id[:10],palette='hls')
plt.xlabel('Customer ID')
plt.title('No.of rides requested')
plt.savefig('plot_7.png')
plt.show()

dist_new=no_dups[no_dups.Distance<30].reset_index(drop=True)
plt.hist(dist_new['Distance'],15, facecolor='blue', alpha=0.5)
plt.title('Histogram of distance covered in rides (Only distances upto 30km are covered)')
plt.xlabel('Distance (km)')
plt.ylabel('Count (No.of rides)')
plt.savefig('plot_8.png')
plt.show()

