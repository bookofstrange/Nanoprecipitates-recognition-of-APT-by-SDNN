# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:20:02 2023

@author: JiWei Yu
"""
#%% generate standard fcc structure
import numpy as np
k0=25
data0=[[i,j,k] for i in range(k0) for j in range(k0) for k in range(k0) ]  
# data4=[[i+0.5,j+0.5,k+0.5] for i in range(k0-1) for j in range(k0-1) for k in range(k0-1) ]
data1=[[i+0.5,j+0.5,k] for i in range(k0-1) for j in range(k0-1) for k in range(k0)]
data2=[[i+0.5,j,k+0.5] for i in range(k0-1) for j in range(k0) for k in range(k0-1)]
data3=[[i,j+0.5,k+0.5] for i in range(k0) for j in range(k0-1) for k in range(k0-1)]
data=np.concatenate((data0,data1,data2,data3))
total_num=len(data)
r_Al=0.926
r_Li=0.030
r_Mg=0.044
num_Al,num_Li,num_Mg=int(total_num*r_Al),int(total_num*r_Li),int(total_num*r_Mg)
np.random.shuffle(data)
atom_Al=data[0:num_Al]
atom_Li,atom_Mg=data[num_Al:num_Al+num_Li],data[num_Al+num_Li:num_Al+num_Li+num_Mg]
atom_Al=np.concatenate((atom_Al,np.zeros((len(atom_Al),1))),axis=-1)
atom_Li=np.concatenate((atom_Li,np.ones((len(atom_Li),1))),axis=-1)
atom_Mg=np.concatenate((atom_Mg,np.ones((len(atom_Mg),1))*2),axis=-1)
data=np.concatenate((atom_Al,atom_Li,atom_Mg),axis=0)
data=np.concatenate((data,np.zeros((len(data),1))),axis=1)
#%% show crystal structure with o3d
import open3d as o3d
points1=atom_Al[:,0:3]
point_cloud1 = o3d.geometry.PointCloud()
point_cloud1.points =o3d.utility.Vector3dVector(points1)    
point_cloud1.paint_uniform_color([0,0,1])
points2=atom_Li[:,0:3]
point_cloud2 = o3d.geometry.PointCloud()
point_cloud2.points =o3d.utility.Vector3dVector(points2)    
point_cloud2.paint_uniform_color([0,1,0])
points3=atom_Mg[:,0:3]
point_cloud3 = o3d.geometry.PointCloud()
point_cloud3.points =o3d.utility.Vector3dVector(points3)    
point_cloud3.paint_uniform_color([1,0,0])
# img=o3d.pybind.visualization.draw_geometries([point_cloud1,point_cloud2,point_cloud3])
# img=o3d.pybind.visualization.draw_geometries([point_cloud2,point_cloud3])
img=o3d.pybind.visualization.draw_geometries([point_cloud2])
#%%
vf=0.2
import random
num_cluster=8
# num_cluster=random.randint()
r=((k0**3*vf)/num_cluster / (4/3*3.14)) ** (1/3)
#%% function with fcc/l12 transition
def L12(x,y,z,r,data):
    x0,y0,z0=x-r,y-r,z-r
    l=r*2+1
    data5=[[x0+i,y0+j,z0+k,1.0,1.0] for i in range(l) for j in range(l-1) for k in range(l-1) ]  
    # data4=[[i+0.5,j+0.5,k+0.5] for i in range(k0-1) for j in range(k0-1) for k in range(k0-1) ]
    data6=[[x0+i+0.5,y0+j+0.5,z0+k,0,1] for i in range(l-1) for j in range(l-1) for k in range(l)]
    data7=[[x0+i+0.5,y0+j,z0+k+0.5,0,1] for i in range(l-1) for j in range(l) for k in range(l-1)]
    data8=[[x0+i,y0+j+0.5,z0+k+0.5,0,1] for i in range(l) for j in range(l-1) for k in range(l-1)]
    
    k=random.uniform(0.2,0.25)
    data5,data6,data7,data8=np.array(data5),np.array(data6),np.array(data7),np.array(data8)
    l_Mg=int(k*len(data5))
    np.random.shuffle(data5)
    data5[:l_Mg,3]=2
    data_new=np.concatenate((data5,data6,data7,data8))
    mark1=((data_new[:,0]-x)**2+(data_new[:,1]-y)**2+(data_new[:,2]-z)**2)<=r**2
    mark3=(0<data_new[:,0])&(data_new[:,0]<k0)&(0<data_new[:,1])&(data_new[:,1]<k0)&(0<data_new[:,2])&(data_new[:,2]<k0)
    data_l12=data_new[mark1&mark3]
    mark2=((data[:,0]-x)**2+(data[:,1]-y)**2+(data[:,2]-z)**2)>r**2
    data_fcc=data[mark2]
    data_f=np.concatenate((data_l12,data_fcc))
    return data_f
def fcc_to_l12(data):
    x,y,z=random.randint(0,k0),random.randint(0,k0),random.randint(0,k0)
    r=random.randint(3,7)
    # Al3Li
    data_f=L12(x,y,z,r,data)
    return data_f
num=random.randint(3,5)
for i in range(num):
    data=fcc_to_l12(data)
atom_Al,atom_Li,atom_Mg=data[data[:,3]==0],data[data[:,3]==1],data[data[:,3]==2]
#%%
import sys
sys.exit()
#%%
def noise(atom,noise=0,noise_z=0,missing=0,t=1):
    flux=np.random.normal(0,noise,((atom.shape[0],3)))
    flux_z=np.random.normal(0,noise_z,((atom.shape[0],3)))

    mark4=(flux[:,0]>t*noise)|(flux[:,0]<-t*noise)
    mark5=(flux[:,1]>t*noise)|(flux[:,1]<-t*noise)
    mark6=(flux[:,2]>t*noise_z)|(flux[:,2]<-t*noise_z)
    flux[mark4,0]=np.random.uniform(-t*noise,t*noise)
    flux[mark5,1]=np.random.uniform(-t*noise,t*noise)
    flux_z[mark6,2]=np.random.uniform(-t*noise_z,t*noise_z)
    atom[:,0]+=flux[:,0]
    atom[:,1]+=flux[:,1]
    atom[:,2]+=flux_z[:,2]
    k5=int(missing*len(atom))
    np.random.shuffle(atom)
    atom=atom[k5:]
    return atom
data=noise(data,1,0.2,0.48,1)
atom_Al,atom_Li,atom_Mg=data[data[:,3]==0],data[data[:,3]==1],data[data[:,3]==2]

