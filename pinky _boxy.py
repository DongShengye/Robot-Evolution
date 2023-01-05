#Import Library
import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram,compileShader
import numpy as np
import pyrr
import math
import matplotlib.pyplot as plt
import time as ti
import copy
import random
import datetime
###########################################################
# [[6.681416010878962], [array([0.73518158, 3.16932885, 1.40127118]), -0.9534456732508954, -0.9561545644593595, 19527.344011385576], [array([ 1.49264708, -0.4905113 ,  1.26364654]), -0.6955978090871906, 0.06327780813700157, 5918.7897254993595], [array([2.83064471, 1.05610337, 0.22932922]), 0.8340993902736138, 0.3282968497864543, 19542.586297218637], [array([0.19142562, 0.5260372 , 0.83405306]), -0.35564206226706985, 0.9198348445203077, 11387.585619959382], [2, 3, 1], [4, 3, 0], [2, 3, 0], [2, 5, 1], [3, 4, 0], [3, 2, 1], [4, 4, 0], [5, 2, 1], [4, 2, 1], [5, 3, 1], [1, 4, 0], [1, 4, 1]]

springs = []
masses = []
avoid_collide = np.zeros([50,50,50]) #important
cubes = []
position_foropengl = []
#constants
k = 10000
g = 10 #m/s^2
m = 60 #kg #80 is standard
a = 1  #global definition

w = 40 #global definition
T = 0
dt = 0.0005
r_p = 100000 #220000
damping =1000   #50000
rest = 1
cube_indices = []
cubey = []
class Mass:
    def __init__(self,mass,indices, acceleration,velocity,id):
        self.indices = indices
        self.mass = mass
        self.velocity = velocity
        self.acceleration = acceleration
        self.id = id

    def __eq__(self, other):
        return isinstance(other, Mass) and self.id == other.id
    def __hash__(self):
        return hash(self.id)   
      
class Spring:
    def __init__(self,restlength,k,idfor2,b,c,coe):
        self.restlength = restlength
        self.k = k
        self.idfor2 = idfor2
        self.id1 = idfor2[0]
        self.id2 = idfor2[1]
        self.coe = coe
        self.b = b
        self.c = c

    def Sforce1(self,indice1,indice2,v1,v2):
        direction_unfinished = np.array(indice1)-np.array(indice2)
        changedlength = np.sqrt(direction_unfinished[0]**2+direction_unfinished[1]**2+direction_unfinished[2]**2)
        direction_finished = direction_unfinished/changedlength
        F_indirection = np.array(self.k*(-changedlength+self.restlength)*direction_finished)-0.15*(v1-v2)*np.array([k,k,k])
        return F_indirection
    
    def Sforce2(self,indice1,indice2,v1,v2):
        direction_unfinished = np.array(indice1)-np.array(indice2)
        changedlength = np.sqrt(direction_unfinished[0]**2+direction_unfinished[1]**2+direction_unfinished[2]**2)
        direction_finished = direction_unfinished/changedlength
        F_indirection = np.array(self.k*(changedlength-self.restlength)*direction_finished)-0.15*(v2-v1)*np.array([k,k,k])
        return F_indirection   
    
    def update_rest(self,t):
        self.restlength = self.coe*(1+self.b*math.sin(w*t+self.c))
        
    def __eq__(self, other):
        return self.idfor2 == tuple(other.idfor2)
    def __hash__(self):
        return hash(('idfor2', tuple(self.idfor2)))  


def meta_grow(initialindices,cubenum):
    start_point_pool = []
    x = initialindices[0]
    y = initialindices[1]
    z = initialindices[2]
    start_point_pool.append([x,y,z])
    avoid_collide[x,y,z] = 1
    cubes.append([x,y,z])
    for i in range(cubenum-1):
        
        status = 1
        while status == 1:
            a = random.randint(1,6)
            match a:
                case 1:
                    if avoid_collide[x+1,y,z] == 1:
                        status = 1
                    else:
                        status = 0                    
                case 2:
                    if avoid_collide[x-1,y,z] == 1:
                        status = 1
                    else:
                        status = 0
                case 3:
                    if avoid_collide[x,y+1,z] == 1:
                        status = 1
                    else:
                        status = 0
                case 4:
                    if avoid_collide[x,y-1,z] == 1:
                        status = 1
                    else:
                        status = 0                  
                case 5:
                    if avoid_collide[x,y,z+1] == 1:
                        status = 1
                    else:
                        status = 0    

        match a:
            case 1:
                avoid_collide[x+1,y,z] = 1
                start_point_pool.append([x+1,y,z])
                buffer = [x+1,y,z]
            case 2:
                avoid_collide[x-1,y,z] = 1
                start_point_pool.append([x-1,y,z])
                buffer = [x-1,y,z]
            case 3:
                avoid_collide[x,y+1,z] = 1
                start_point_pool.append([x,y+1,z])
                buffer = [x,y+1,z]
            case 4:
                avoid_collide[x,y-1,z] = 1 
                start_point_pool.append([x,y-1,z]) 
                buffer = [x,y-1,z]                
            case 5:
                avoid_collide[x,y,z+1] = 1 
                start_point_pool.append([x,y,z+1])  
                buffer = [x,y,z+1]    
        qwe = random.randint(0,len(start_point_pool)-1)
        x = start_point_pool[qwe][0]
        y = start_point_pool[qwe][1]
        z = start_point_pool[qwe][2]
        cubes.append([buffer[0],buffer[1],buffer[2]])
    return cubes

def cubemaker(cubes, length): #q is th coordinates
    
    x = cubes[0][0]
    y = cubes[0][1]
    z = cubes[0][2]
    
    
    masses.append(Mass(m,np.array([x,y,z]),np.array([0,0,0]),np.array([0.0000001,0.0000001,0.0000001]),0))
    masses.append(Mass(m,np.array([x-length,y,z]),np.array([0,0,0]),np.array([0.0000001,0.0000001,0.0000001]),1))
    masses.append(Mass(m,np.array([x-length,y-length,z]),np.array([0,0,0]),np.array([0.0000001,0.0000001,0.0000001]),2))
    masses.append(Mass(m,np.array([x,y-length,z]),np.array([0,0,0]),np.array([0.0000001,0.0000001,0.0000001]),3))
    masses.append(Mass(m,np.array([x,y,z+length]),np.array([0,0,0]),np.array([0.0000001,0.0000001,0.0000001]),4))
    masses.append(Mass(m,np.array([x-length,y,z+length]),np.array([0,0,0]),np.array([0.0000001,0.0000001,0.0000001]),5))
    masses.append(Mass(m,np.array([x-length,y-length,z+length]),np.array([0,0,0]),np.array([0.0000001,0.0000001,0.0000001]),6))
    masses.append(Mass(m,np.array([x,y-length,z+length]),np.array([0,0,0]),np.array([0.0000001,0.0000001,0.0000001]),7))    
    cube_indices.append([0,1,2,3,4,5,6,7])
    springs.append(Spring(rest,k,[0,1],0,0,1))
    springs.append(Spring(rest,k,[1,2],0,0,1))
    springs.append(Spring(rest,k,[2,3],0,0,1))
    springs.append(Spring(rest,k,[3,0],0,0,1))
    
    springs.append(Spring(rest,k,[4,5],0,0,1))
    springs.append(Spring(rest,k,[5,6],0,0,1))
    springs.append(Spring(rest,k,[6,7],0,0,1))
    springs.append(Spring(rest,k,[7,4],0,0,1)) 
    
    springs.append(Spring(rest,k,[0,4],0,0,1))
    springs.append(Spring(rest,k,[1,5],0,0,1))
    springs.append(Spring(rest,k,[2,6],0,0,1))
    springs.append(Spring(rest,k,[3,7],0,0,1))        

    springs.append(Spring(math.sqrt(3)*rest,k,[0,6],0,0,math.sqrt(3)))
    springs.append(Spring(math.sqrt(3)*rest,k,[1,7],0,0,math.sqrt(3)))
    springs.append(Spring(math.sqrt(3)*rest,k,[2,4],0,0,math.sqrt(3)))
    springs.append(Spring(math.sqrt(3)*rest,k,[3,5],0,0,math.sqrt(3)))
    
    springs.append(Spring(math.sqrt(2)*rest,k,[0,7],0,0,math.sqrt(2)))
    springs.append(Spring(math.sqrt(2)*rest,k,[3,4],0,0,math.sqrt(2)))  
    
    springs.append(Spring(math.sqrt(2)*rest,k,[0,5],0,0,math.sqrt(2)))
    springs.append(Spring(math.sqrt(2)*rest,k,[1,4],0,0,math.sqrt(2)))  
    
    springs.append(Spring(math.sqrt(2)*rest,k,[1,6],0,0,math.sqrt(2)))
    springs.append(Spring(math.sqrt(2)*rest,k,[2,5],0,0,math.sqrt(2))) 
    
    springs.append(Spring(math.sqrt(2)*rest,k,[3,6],0,0,math.sqrt(2)))
    springs.append(Spring(math.sqrt(2)*rest,k,[2,7],0,0,math.sqrt(2)))
    
    springs.append(Spring(math.sqrt(2)*rest,k,[4,6],0,0,math.sqrt(2)))
    springs.append(Spring(math.sqrt(2)*rest,k,[5,7],0,0,math.sqrt(2)))   
    
    springs.append(Spring(math.sqrt(2)*rest,k,[0,2],0,0,math.sqrt(2)))
    springs.append(Spring(math.sqrt(2)*rest,k,[1,3],0,0,math.sqrt(2)))  
    maindex = 7
    cubey.append([0,1,2,3,4,5,6,7])
    for i in range(len(cubes)-1):

        x = cubes[i+1][0]
        y = cubes[i+1][1]
        z = cubes[i+1][2]

        m1=m2=m3=m4=m5=m6=m7=m8= None


  
        for i in range(len(masses)):
            if (masses[i].indices == np.array([x,y,z])).all():
                m1 = masses[i].id                    
            if (masses[i].indices == np.array([x-length,y,z])).all():
                m2 = masses[i].id
            if (masses[i].indices == np.array([x-length,y-length,z])).all():
                m3 = masses[i].id                    
            if (masses[i].indices == np.array([x,y-length,z])).all():
                m4 = masses[i].id                    

            if (masses[i].indices == np.array([x,y,z+length])).all():
                m5 = masses[i].id                    
            if (masses[i].indices == np.array([x-length,y,z+length])).all():
                m6 = masses[i].id
            if (masses[i].indices == np.array([x-length,y-length,z+length])).all():
                m7 = masses[i].id                    
            if (masses[i].indices == np.array([x,y-length,z+length])).all():
                m8 = masses[i].id       
  
        if m1==None:
            m1 = maindex+1
            masses.append(Mass(m,np.array([x,y,z]),np.array([0,0,0]),np.array([0.0000001,0.0000001,0.0000001]),m1)) #m1
        if m2==None:
            m2 = maindex+2
            masses.append(Mass(m,np.array([x-length,y,z]),np.array([0,0,0]),np.array([0.0000001,0.0000001,0.0000001]),m2)) #m2
        if m3==None:
            m3 = maindex+3
            masses.append(Mass(m,np.array([x-length,y-length,z]),np.array([0,0,0]),np.array([0.0000001,0.0000001,0.0000001]),m3)) #m3            
        if m4==None:
            m4 = maindex+4
            masses.append(Mass(m,np.array([x,y-length,z]),np.array([0,0,0]),np.array([0.0000001,0.0000001,0.0000001]),m4)) #m4           
        if m5==None:
            m5 = maindex+5
            masses.append(Mass(m,np.array([x,y,z+length]),np.array([0,0,0]),np.array([0.0000001,0.0000001,0.0000001]),m5)) #m5           
        if m6==None:
            m6 = maindex+6
            masses.append(Mass(m,np.array([x-length,y,z+length]),np.array([0,0,0]),np.array([0.0000001,0.0000001,0.0000001]),m6)) #m6           
        if m7==None:
            m7 = maindex+7
            masses.append(Mass(m,np.array([x-length,y-length,z+length]),np.array([0,0,0]),np.array([0.0000001,0.0000001,0.0000001]),m7)) #m1  
        if m8==None:
            m8 = maindex+8
            masses.append(Mass(m,np.array([x,y-length,z+length]),np.array([0,0,0]),np.array([0.0000001,0.0000001,0.0000001]),m8)) #m1                      

        cubey.append([m1,m2,m3,m4,m5,m6,m7,m8])
        
        springs.append(Spring(rest,k,[m1,m2],0,0,1))
        springs.append(Spring(rest,k,[m2,m3],0,0,1))
        springs.append(Spring(rest,k,[m3,m4],0,0,1))
        springs.append(Spring(rest,k,[m4,m1],0,0,1))
        
        
        springs.append(Spring(rest,k,[m5,m6],0,0,1))
        springs.append(Spring(rest,k,[m6,m7],0,0,1))
        springs.append(Spring(rest,k,[m7,m8],0,0,1))
        springs.append(Spring(rest,k,[m8,m5],0,0,1))
        
        springs.append(Spring(rest,k,[m1,m5],0,0,1))
        springs.append(Spring(rest,k,[m2,m6],0,0,1))       
        springs.append(Spring(rest,k,[m3,m7],0,0,1))
        springs.append(Spring(rest,k,[m4,m8],0,0,1))
        
        springs.append(Spring(math.sqrt(3)*rest,k,[m1,m7],0,0,math.sqrt(3)))
        springs.append(Spring(math.sqrt(3)*rest,k,[m2,m8],0,0,math.sqrt(3)))
        springs.append(Spring(math.sqrt(3)*rest,k,[m3,m5],0,0,math.sqrt(3)))
        springs.append(Spring(math.sqrt(3)*rest,k,[m4,m6],0,0,math.sqrt(3)))
        
        springs.append(Spring(math.sqrt(2)*rest,k,[m1,m6],0,0,math.sqrt(2)))
        springs.append(Spring(math.sqrt(2)*rest,k,[m2,m5],0,0,math.sqrt(2)))
        
        springs.append(Spring(math.sqrt(2)*rest,k,[m2,m7],0,0,math.sqrt(2)))
        springs.append(Spring(math.sqrt(2)*rest,k,[m3,m6],0,0,math.sqrt(2)))
        
        springs.append(Spring(math.sqrt(2)*rest,k,[m4,m7],0,0,math.sqrt(2)))
        springs.append(Spring(math.sqrt(2)*rest,k,[m3,m8],0,0,math.sqrt(2)))
        
        springs.append(Spring(math.sqrt(2)*rest,k,[m4,m5],0,0,math.sqrt(2)))
        springs.append(Spring(math.sqrt(2)*rest,k,[m1,m8],0,0,math.sqrt(2)))
        
        springs.append(Spring(math.sqrt(2)*rest,k,[m2,m4],0,0,math.sqrt(2)))
        springs.append(Spring(math.sqrt(2)*rest,k,[m1,m3],0,0,math.sqrt(2)))
        
        springs.append(Spring(math.sqrt(2)*rest,k,[m5,m7],0,0,math.sqrt(2)))
        springs.append(Spring(math.sqrt(2)*rest,k,[m6,m8],0,0,math.sqrt(2)))                
        #if m1==m2==m3==m4==m5==m6==m7==m8==None: 

        cube_indices.append([m1,m2,m3,m4,m5,m6,m7,m8])
        maindex +=8     
    y_min = masses[0].indices[0]
    x_min = masses[0].indices[1]
    z_min = masses[0].indices[2]
    y_max = masses[0].indices[0]
    x_max = masses[0].indices[1]
    z_max = masses[0].indices[2]    
    geno = []
    for i in range(len(masses)):
        if x_min>masses[i].indices[0]:
            x_min = masses[i].indices[0]
        if y_min>masses[i].indices[1]:
            y_min = masses[i].indices[1]      
        if z_min>masses[i].indices[2]:
            z_min = masses[i].indices[2]  
        if x_max<masses[i].indices[0]:
            x_max = masses[i].indices[0]
        if y_max<masses[i].indices[1]:
            y_max = masses[i].indices[1]      
        if z_max<masses[i].indices[2]:
            z_max = masses[i].indices[2]     
    return([x_max,x_min,y_max,y_min,z_max,z_min])
def springkiller(springs):
    springs_new = []
    for i in range(len(springs)):    
        springs[i].idfor2.sort()
        springs[i].idfor2 = tuple(springs[i].idfor2)
    springs_new = list(set(springs))
    return springs_new                                        
def gene_creation(range_1):
    x_max = range_1[0]
    x_min = range_1[1]
    y_max = range_1[2]
    y_min = range_1[3]
    z_max = range_1[4]
    z_min = range_1[5]  
    geno = []

    setup = random.randint(1,6)
    
    for i in range(0,5):     #4 tpyes of materials
        x_cord = random.uniform(x_min,x_max)        
        y_cord = random.uniform(y_min,y_max) 
        z_cord = random.uniform(z_min,z_max)  
        b_cord = random.uniform(-1,1)
        
        
        c_cord = random.uniform(-1,1)
        k = random.uniform(1000,20000)  #krange1
        geno.append([np.array([x_cord,y_cord,z_cord]),b_cord,c_cord,k])
        
    return geno
def spring_assginer(cubey,gene):
    for i in range(len(cubey)):
        #print(cubey[i])
        weird_buffer = copy.deepcopy(cubey[i])
        boxingclub = []
        for q in range(len(masses)):
            if masses[q].id == cubey[i][0]:
                indiA = masses[q].indices
            if masses[q].id == cubey[i][6]:
                indiB = masses[q].indices
                
        for i in range(len(gene)):

            boxingclub.append(math.sqrt(np.sum(((indiA+indiB)/2-gene[i][0])**2)))
        index_for_gene = np.argmin(np.array(boxingclub)) 
        
        b = gene[index_for_gene][1]          
        c = gene[index_for_gene][2] 
        k = gene[index_for_gene][3] 
        #print(b,c,k)
        a = 1
        qwe = 0
        #print(len(springs_new))
        for w in range(len(springs_new)):
            #print("before",springs_new[w].idfor2)
            if set(springs_new[w].idfor2).issubset(weird_buffer):
                #print(springs_new[w].idfor2)
                springs_new[w].a = a
                springs_new[w].b = b
                springs_new[w].c = c
                springs_new[w].k = k
     
                            
        


cubes = meta_grow([3,3,0],20)

create_range = cubemaker([[3, 3, 0], [3, 3, 1], [2, 3, 0], [1, 3, 0], [1, 3, 1], [4, 2, 0], [4, 2, 0], [2, 2, 0], [4, 3, 1], [3, 2, 0], [2, 3, 1], [5, 3, 1]] , a)
#[[3, 3, 0], [4, 3, 0],  [3, 1, 0], [1, 2, 0],[1, 2, 1], [4, 3, 1], [2, 2, 1], [4, 1, 0], [4, 1, 1], [2, 2, 0]] 2.86m/s
#[3, 3, 0], [4, 3, 0], [2, 4, 0], [3, 1, 0], [1, 2, 0], [4, 3, 1], [2, 2, 1], [4, 1, 0], [5, 1, 1], [2, 2, 0] 2.95m/s
# for i in range(len(masses)):
#     if masses[i].id == 8 or masses[i].id == 11 or masses[i].id == 12 or masses[i].id == 15 or masses[i].id == 19 or masses[i].id == 23: #quickaccess
#         masses[i].mass = 100000000000

# [0.6423674619006414, [[1.49797508, 3.94126657, 0.24371293], -0.454182263644288, 0.4862890890576206, 14682.436381994487], [[3.69280641, 1.13395274, 1.71165422], 0.25006781661625244, -0.9816061378378158, 19409.313936755534], [[3.39127151, 2.42057456, 1.93976029], 0.9377025460265247, -0.02256995343183184, 19883.663798665468], [[1.37935555, 1.35807895, 0.15476965], 0.9546267446012014, 0.9575265242660906, 15301.906129683923], [3, 3, 0], [3, 3, 1], [2, 3, 0], [4, 3, 1], [5, 3, 2], [4, 3, 0], [5, 3, 1], [6, 3, 2], [1, 3, 0], [2, 3, 1]] 
#[[2, 3, 1], [4, 3, 0], [2, 3, 0], [2, 5, 1], [3, 4, 0], [3, 2, 1], [4, 4, 0], [5, 2, 1], [4, 2, 1], [5, 3, 1], [1, 4, 0], [1, 4, 1]] ---best
springs_new = springkiller(springs)

# for i in range(len(springs_new)):
#     print("set",springs_new[i].idfor2)
gene = gene_creation(create_range)
# [[[5.81039958, 3.48311336, 0.42134788], -0.48545494121547916, -0.7230769816743177, 15606.287092982398], [[5.02521669, 3.8520274 , 0.21833226], 0.07446527303005857, 0.4470629413341636, 4443.353275875261], [[3.56790396, 2.15641197, 0.05759346], -0.596215165381619, 0.7017714273285409, 19527.80380123405], [[5.36082883, 3.10246834, 0.89705392], -0.6451911817781095, 0.3186639345737863, 11822.34748598118]]]
cubey = np.array(cubey)
#print(cubey)
#print(len(springs_new))
backup_springs = copy.deepcopy(springs_new)
backup_masses = copy.deepcopy(masses)
#[[6.681416010878962], [[0.73518158, 3.16932885, 1.40127118], -0.9534456732508954, -0.9561545644593595, 19527.344011385576], [[ 1.49264708, -0.4905113 ,  1.26364654], -0.6955978090871906, 0.06327780813700157, 5918.7897254993595], [[2.83064471, 1.05610337, 0.22932922], 0.8340993902736138, 0.3282968497864543, 19542.586297218637], [[0.19142562, 0.5260372 , 0.83405306], -0.35564206226706985, 0.9198348445203077, 11387.585619959382]]
spring_assginer(cubey,[[[-19.14743413,  -6.54286013,  -0.64795811], 0.7298928445729047, 0.5314368142355428, 2605.55810667575], [[1.90847401, 1.57796916, 2.76330674], 0.979918002746208, 0.7401533787367103, 19720.104377699216], [[-2.55472417, -2.92125809, -2.33300682], 0.7059022988765935, 0.5489367011995706, 4442.492774805447], [[0.83792921, 2.62790736, 1.45726644], -0.8239402101174411, -0.9348395007144714, 17207.716874531936]])
#[[1.7203552 , 2.27133355, 1.61605802], -0.9829858218211891, -0.9190971718307764, 30716.41847244401], [[19.14678719, -1.60772655,  2.50577888], 0.14113377463170462, -0.9591638822060504, -244.61274352270132], [[4.14146336, 2.26501614, 1.61188451], 1.1250811788560644, 0.005662390897579893, 25066.09900242173], [[8.30856047, 7.30866663, 1.84462348], 0.05893288194883444, -0.19264229244531092, 18157.83626760393], [3, 3, 0], [4, 3, 0], [2, 4, 0], [3, 1, 0], [1, 2, 0], [4, 3, 1], [2, 2, 1], [4, 1, 0], [4, 1, 1], [2, 1, 0]] 
#[151.94031459097562, [[1.7203552 , 2.27133355, 1.61605802], -0.9829858218211891, 0.5077196146788328, -26013.928020968873], [[42.87414378, -3.60007654,  5.61102616], 0.007330257946800334, -0.9591638822060504, -214.69498159333273], [[-8.76844614, -4.79556869, -3.41273633], -0.4372504951913718, 2.7353766589594566e-06, 29286.298175904103], [[-48.9262457 , -43.03821589, -10.86235117], -0.005450735088554715, 0.23028453534935803, 11366.117583933337], [3, 3, 1], [3, 2, 0], [3, 4, 0], [4, 3, 1], [4, 2, 1], [4, 4, 1], [3, 1, 1], [4, 0, 1], [4, 1, 1], [3, 0, 0]] 
#[110.8515478216822, [[1.7203552 , 2.27133355, 1.61605802], -0.9829858218211891, 0.906587480660555, -26013.928020968873], [[42.87414378, -3.60007654,  5.61102616], 0.007330257946800334, -0.9591638822060504, 183.32147668124293], [[-8.76844614, -4.79556869, -3.41273633], 1.2666506822261219, 2.7353766589594566e-06, 29286.298175904103], [[-48.9262457 , -43.03821589, -10.86235117], -0.019966552432265913, 0.23028453534935803, 11366.117583933337], [3, 3, 1], [3, 2, 0], [3, 4, 0], [4, 3, 1], [4, 2, 1], [4, 4, 1], [3, 2, 1], [3, 0, 1], [4, 1, 1], [3, 0, 0]] 
#[0.5212104470231046, [[3.89973742, 2.87931029, 1.58422361], 0.9236103377969442, 0.17101908799772736, 8456.952666389865], [[2.08916691, 3.54556801, 0.6934616 ], -0.9944139611505243, -0.21180944006854818, 14145.873495378808], [[3.07796679, 2.37146745, 1.4406168 ], 0.7278212800595145, 0.44082675129267157, 15645.571263858334], [[2.57976713, 2.54069051, 1.27789279], -0.6582280683101966, -0.7187683764936676, 18958.003995064926], [3, 3, 0], [2, 3, 0], [2, 3, 1], [4, 3, 0], [2, 2, 0], [4, 3, 1], [1, 3, 0], [1, 3, 1], [3, 3, 1], [2, 2, 0]] 
#0.5213404830204039, [[5.68738562, 3.47000197, 1.68987583], 0.010262333806127582, -0.05169860648608874, 17044.336684007103], [[2.9435013 , 3.57602647, 1.49418066], -0.7157043504406938, -0.2057580198625628, 20392.70029999729], [[4.22257068, 3.41834135, 0.85661428], 0.888782123747289, 0.9145335391775855, 13696.073089128642], [[6.02609781, 7.38560426, 0.97894062], 0.6737099974986951, 0.1981146580555071, 16953.86274896603], [3, 3, 0], [4, 3, 0], [2, 4, 0], [4, 2, 0], [2, 4, 1], [1, 4, 1], [1, 4, 0], [1, 3, 0], [4, 4, 0], [4, 4, 1], [2, 3, 0], [3, 4, 1], [5, 3, 1], [4, 2, 0]] 

#[0.5224646115554049, [[5.68738562, 3.47000197, 1.68987583], 0.010262333806127582, -0.05169860648608874, 17044.336684007103], [[2.9435013 , 3.57602647, 1.49418066], -0.7157043504406938, -0.2057580198625628, 20392.70029999729], [[4.22257068, 3.41834135, 0.85661428], 0.888782123747289, 0.9145335391775855, 13696.073089128642], [[6.02609781, 7.38560426, 0.97894062], 0.6737099974986951, 0.1981146580555071, 16953.86274896603], [3, 3, 0], [4, 3, 0], [2, 4, 0], [4, 2, 0], [3, 4, 1], [1, 4, 1], [1, 4, 0], [1, 3, 0], [4, 4, 0], [4, 4, 2], [2, 3, 0], [3, 5, 1], [5, 4, 1], [4, 2, 0]] 
#[0.6024787073759766, [[1.49797508, 3.94126657, 0.24371293], 0.9933511133077431, 0.4862890890576206, 14682.436381994487], [[3.69280641, 1.13395274, 1.71165422], 0.25006781661625244, -0.9816061378378158, 19409.313936755534], [[3.39127151, 2.42057456, 1.93976029], 0.9377025460265247, -0.02885630613878809, 19444.26581173342], [[1.37935555, 1.35807895, 0.15476965], 0.9206046886222343, 0.9575265242660906, 15301.906129683923], [3, 3, 0], [3, 3, 1], [2, 3, 0], [4, 3, 1], [4, 3, 2], [4, 3, 0], [5, 3, 2], [5, 3, 1], [1, 3, 0], [2, 3, 1]] 
#[0.6487013459118217, [[-19.14743413,  -6.54286013,  -0.64795811], 0.7298928445729047, 0.5314368142355428, 2605.55810667575], [[1.90847401, 1.57796916, 2.76330674], 0.979918002746208, 0.7401533787367103, 19720.104377699216], [[-2.55472417, -2.92125809, -2.33300682], 0.7059022988765935, 0.5489367011995706, 4442.492774805447], [[0.83792921, 2.62790736, 1.45726644], -0.8239402101174411, -0.9348395007144714, 17207.716874531936], [3, 3, 0], [3, 3, 1], [2, 3, 0], [1, 3, 0], [1, 3, 1], [4, 2, 0], [4, 2, 0], [2, 2, 0], [4, 3, 1], [3, 2, 0], [2, 3, 1], [5, 3, 1]] 
class Cube:


    def __init__(self, position, eulers):

        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)

class App:
    
    def __init__(self,w):
        
        self.w = w
        #initialise pygame
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)#set the using version of OpenGL
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)#set the using version of OpenGL
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK,
                                    pg.GL_CONTEXT_PROFILE_CORE)#disable deprecated features
        pg.display.set_mode((1920,1080), pg.OPENGL|pg.DOUBLEBUF)#set up the display window -- important
        self.clock = pg.time.Clock()
        #self.cube_mesh = CubeMesh
        self.mainLoop(self.w)

    
    def mainLoop(self,w):


        a = 1 #can't be changed
        current_time = datetime.datetime.now()
        initialminute=current_time.minute
        initialsec = current_time.second
        "Second :", current_time.second
        #w = 1
        Us = 0.5
        Uk = 0.2
        indi1 = []
        indi2 = []

        p_subset = []
        
        
        Fitness = 0
        Original_fitness = 0
        running = True


        format = []
        for i in range(len(masses)):
            Original_fitness = Original_fitness+masses[i].indices[0]
            format.append([])
        for i in range(len(masses)):
            format[i].append(masses[i].id) 
            
        running = True
        t = -dt
        while (running):
            #cube_coordinate = [None]*(12*15*len(cubey))
            cube_coordinate = []
            
            t = t+dt
            #check events
            for event in pg.event.get():
                if (event.type == pg.QUIT):
                    running = False
            if t >= 4:
                running = False 
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


            Ftotal_store = []
            force_formass = copy.deepcopy(format)
            mass_spring_assign = []

            
            for i in range(len(springs_new)): 
                for e in range(len(masses)):
                    if springs_new[i].idfor2[0] == masses[e].id:
                        indi1 = masses[e].indices
                        v1 = masses[e].velocity
                    if springs_new[i].idfor2[1] == masses[e].id: 
                        indi2 = masses[e].indices   
                        v2 = masses[e].velocity
                springs_new[i].update_rest(t)      
                for r in range(len(force_formass)):
                    if force_formass[r][0] == springs_new[i].idfor2[0]:
                        force_formass[r].append(springs_new[i].Sforce1(indi1,indi2,v1,v2))
                    if force_formass[r][0] == springs_new[i].idfor2[1]:
                        force_formass[r].append(springs_new[i].Sforce2(indi1,indi2,v1,v2))

            for i in range(len(masses)):
                
                sumed_forces = np.sum(force_formass[i][1:],axis = 0)
                
                if masses[i].indices[2]>0:

                    Ftotal = sumed_forces+np.array([0,0,-m*g])
                    #print(Ftotal)

                else:
                    
                    FN = sumed_forces[2]-r_p*masses[i].indices[2]-m*g
                    #print(FN)
                    FH = math.sqrt(sumed_forces[0]**2+sumed_forces[1]**2)
                    
                    if FH<-FN*Us:
                        Ftotal = sumed_forces+np.array([0,0,-m*g])+np.array([0,0,-r_p*masses[i].indices[2]])

                    if FH>=-FN*Us:
                        
                        # denominator = math.sqrt(sumed_forces[0]**2+sumed_forces[1]**2)
                        # Ftotal = sumed_forces-np.array([sumed_forces[0]/denominator*(Uk*FN),sumed_forces[1]/denominator*(Uk*FN),0])+np.array([0,0,-m*g])+np.array([0,0,-r_p*masses[i].indices[2]])
                        denominator = math.sqrt(masses[i].velocity[0]**2+masses[i].velocity[1]**2)
                        
                        Ftotal = sumed_forces-np.array([masses[i].velocity[0]/denominator*(Uk*FN),masses[i].velocity[1]/denominator*(Uk*FN),0])+np.array([0,0,-m*g])+np.array([0,0,-r_p*masses[i].indices[2]])#-damping*masses[i].velocity
                Ftotal_store.append(Ftotal)
            Fitness = 0 
            
            for i in range(len(masses)):
                masses[i].acceleration = Ftotal_store[i]/m
                #print("before",masses[i].velocity[2])
                masses[i].velocity = 1*(masses[i].velocity+masses[i].acceleration*dt) #dampingop2   0.95*
                #print(masses[i].velocity[2])
                masses[i].indices = masses[i].indices+masses[i].velocity*dt
                if running == False:
                    Fitness += masses[i].indices[0]
                    current_time = datetime.datetime.now()
                    finalminute=current_time.minute
                    finalsec = current_time.second

            for i in range(len(cubey)):
                p_subset = []
                for w in range(len(cubey[0])):
                    for q in range(len(masses)):
                        if cubey[i,w] == masses[q].id:
                            p_subset.append(masses[q].indices)
                
                cube_coordinate.append(p_subset[0][0])
                cube_coordinate.append(p_subset[0][1])
                cube_coordinate.append(p_subset[0][2])
                cube_coordinate.append(0)       
                cube_coordinate.append(0)
                cube_coordinate.append(p_subset[1][0])
                cube_coordinate.append(p_subset[1][1])
                cube_coordinate.append(p_subset[1][2])
                cube_coordinate.append(0)       
                cube_coordinate.append(1)            
                cube_coordinate.append(p_subset[2][0]) 
                cube_coordinate.append(p_subset[2][1]) 
                cube_coordinate.append(p_subset[2][2]) 
                cube_coordinate.append(1)       
                cube_coordinate.append(1) 
                
                cube_coordinate.append(p_subset[2][0])
                cube_coordinate.append(p_subset[2][1])
                cube_coordinate.append(p_subset[2][2])
                cube_coordinate.append(0)       
                cube_coordinate.append(0)            
                cube_coordinate.append(p_subset[3][0])
                cube_coordinate.append(p_subset[3][1])
                cube_coordinate.append(p_subset[3][2])
                cube_coordinate.append(0)       
                cube_coordinate.append(1)            
                cube_coordinate.append(p_subset[0][0]) 
                cube_coordinate.append(p_subset[0][1]) 
                cube_coordinate.append(p_subset[0][2]) 
                cube_coordinate.append(1)       
                cube_coordinate.append(1) 
                
                cube_coordinate.append(p_subset[4][0])
                cube_coordinate.append(p_subset[4][1])
                cube_coordinate.append(p_subset[4][2])
                cube_coordinate.append(0)       
                cube_coordinate.append(0)            
                cube_coordinate.append(p_subset[5][0])
                cube_coordinate.append(p_subset[5][1])
                cube_coordinate.append(p_subset[5][2])
                cube_coordinate.append(0)       
                cube_coordinate.append(1)            
                cube_coordinate.append(p_subset[6][0]) 
                cube_coordinate.append(p_subset[6][1])
                cube_coordinate.append(p_subset[6][2])
                cube_coordinate.append(1)       
                cube_coordinate.append(1)                         
                
                cube_coordinate.append(p_subset[6][0])
                cube_coordinate.append(p_subset[6][1])
                cube_coordinate.append(p_subset[6][2])
                cube_coordinate.append(0)       
                cube_coordinate.append(0)            
                cube_coordinate.append(p_subset[7][0])
                cube_coordinate.append(p_subset[7][1])
                cube_coordinate.append(p_subset[7][2])
                cube_coordinate.append(0)       
                cube_coordinate.append(1)            
                cube_coordinate.append(p_subset[4][0]) 
                cube_coordinate.append(p_subset[4][1])
                cube_coordinate.append(p_subset[4][2])
                cube_coordinate.append(1)       
                cube_coordinate.append(1)   
                
                cube_coordinate.append(p_subset[3][0])
                cube_coordinate.append(p_subset[3][1])
                cube_coordinate.append(p_subset[3][2])
                cube_coordinate.append(0)       
                cube_coordinate.append(0)            
                cube_coordinate.append(p_subset[0][0])
                cube_coordinate.append(p_subset[0][1])
                cube_coordinate.append(p_subset[0][2])
                cube_coordinate.append(0)       
                cube_coordinate.append(1)            
                cube_coordinate.append(p_subset[4][0]) 
                cube_coordinate.append(p_subset[4][1]) 
                cube_coordinate.append(p_subset[4][2]) 
                cube_coordinate.append(1)       
                cube_coordinate.append(1)    
                
                cube_coordinate.append(p_subset[3][0])
                cube_coordinate.append(p_subset[3][1])
                cube_coordinate.append(p_subset[3][2])
                cube_coordinate.append(0)       
                cube_coordinate.append(0)            
                cube_coordinate.append(p_subset[7][0])
                cube_coordinate.append(p_subset[7][1])
                cube_coordinate.append(p_subset[7][2])
                cube_coordinate.append(0)       
                cube_coordinate.append(1)            
                cube_coordinate.append(p_subset[4][0]) 
                cube_coordinate.append(p_subset[4][1]) 
                cube_coordinate.append(p_subset[4][2]) 
                cube_coordinate.append(1)       
                cube_coordinate.append(1) 
                
                cube_coordinate.append(p_subset[4][0])
                cube_coordinate.append(p_subset[4][1])
                cube_coordinate.append(p_subset[4][2])
                cube_coordinate.append(0)       
                cube_coordinate.append(0)            
                cube_coordinate.append(p_subset[0][0])
                cube_coordinate.append(p_subset[0][1])
                cube_coordinate.append(p_subset[0][2])
                cube_coordinate.append(0)       
                cube_coordinate.append(1)            
                cube_coordinate.append(p_subset[1][0]) 
                cube_coordinate.append(p_subset[1][1]) 
                cube_coordinate.append(p_subset[1][2]) 
                cube_coordinate.append(1)       
                cube_coordinate.append(1) 
                
                cube_coordinate.append(p_subset[4][0])
                cube_coordinate.append(p_subset[4][1])
                cube_coordinate.append(p_subset[4][2])
                cube_coordinate.append(0)       
                cube_coordinate.append(0)            
                cube_coordinate.append(p_subset[5][0])
                cube_coordinate.append(p_subset[5][1])
                cube_coordinate.append(p_subset[5][2])
                cube_coordinate.append(0)       
                cube_coordinate.append(1)            
                cube_coordinate.append(p_subset[1][0]) 
                cube_coordinate.append(p_subset[1][1]) 
                cube_coordinate.append(p_subset[1][2]) 
                cube_coordinate.append(1)       
                cube_coordinate.append(1) 
                
                
                cube_coordinate.append(p_subset[1][0])
                cube_coordinate.append(p_subset[1][1])
                cube_coordinate.append(p_subset[1][2])
                cube_coordinate.append(0)       
                cube_coordinate.append(0)            
                cube_coordinate.append(p_subset[5][0])
                cube_coordinate.append(p_subset[5][1])
                cube_coordinate.append(p_subset[5][2])
                cube_coordinate.append(0)       
                cube_coordinate.append(1)            
                cube_coordinate.append(p_subset[6][0])
                cube_coordinate.append(p_subset[6][1])
                cube_coordinate.append(p_subset[6][2]) 
                cube_coordinate.append(1)       
                cube_coordinate.append(1) 
                

                cube_coordinate.append(p_subset[1][0])
                cube_coordinate.append(p_subset[1][1])
                cube_coordinate.append(p_subset[1][2])                
                cube_coordinate.append(0)       
                cube_coordinate.append(0)            
                cube_coordinate.append(p_subset[2][0])
                cube_coordinate.append(p_subset[2][1])
                cube_coordinate.append(p_subset[2][2])                
                cube_coordinate.append(0)       
                cube_coordinate.append(1)            
                cube_coordinate.append(p_subset[6][0])
                cube_coordinate.append(p_subset[6][1])
                cube_coordinate.append(p_subset[6][2])                 
                cube_coordinate.append(1)       
                cube_coordinate.append(1)   
                
                

                cube_coordinate.append(p_subset[2][0])
                cube_coordinate.append(p_subset[2][1])
                cube_coordinate.append(p_subset[2][2])                
                cube_coordinate.append(0)       
                cube_coordinate.append(0)            
                cube_coordinate.append(p_subset[3][0])
                cube_coordinate.append(p_subset[3][1])
                cube_coordinate.append(p_subset[3][2])                
                cube_coordinate.append(0)       
                cube_coordinate.append(1)             
                cube_coordinate.append(p_subset[7][0])
                cube_coordinate.append(p_subset[7][1])
                cube_coordinate.append(p_subset[7][2])                
                cube_coordinate.append(1)       
                cube_coordinate.append(1) 
                

                cube_coordinate.append(p_subset[2][0])
                cube_coordinate.append(p_subset[2][1])
                cube_coordinate.append(p_subset[2][2])                
                cube_coordinate.append(0)       
                cube_coordinate.append(0)            
                cube_coordinate.append(p_subset[6][0])
                cube_coordinate.append(p_subset[6][1])
                cube_coordinate.append(p_subset[6][2])                
                cube_coordinate.append(0)       
                cube_coordinate.append(1)            
 
                cube_coordinate.append(p_subset[7][0])
                cube_coordinate.append(p_subset[7][1])
                cube_coordinate.append(p_subset[7][2])                
                cube_coordinate.append(1)       
                cube_coordinate.append(1) 

            
            Rendered = RenderPass(cube_coordinate)
            
            Rendered.render()
          
            pg.display.flip()

            #timing
            #self.clock.tick(300)

            
            
            Rendered.cube_mesh.destroy()
            Rendered.grid_mesh.destroy()
            Rendered.wood_texture.destroy()
            Rendered.black_texture.destroy()
        fitty = Fitness-Original_fitness
        print("fitness",fitty)
        print('use the real time:',finalminute-initialminute,'minutes',finalsec-initialsec,'seconds')
        print(f'virtual time {t}')
        print('masscount',len(masses))
        print(f'The velocity of this robot is {fitty/len(masses)/t}')
        self.quit(Rendered.shader)
        
    def quit(self,shader):
        glDeleteProgram(shader)
        pg.quit()

class RenderPass:

    def __init__(self,cube_coordinate):
        #initialise opengl
        glClearColor(0.1, 0.2, 0.2, 1) #0.1, 0.2, 0.2, 1
        self.cube_coordinate = cube_coordinate
        self.shader = self.createShader("shaders/vertex.txt", "shaders/fragment.txt")
        glUseProgram(self.shader)
        glUniform1i(glGetUniformLocation(self.shader, "imageTexture"), 0)
        glEnable(GL_DEPTH_TEST)

        self.wood_texture = Material("gfx/blue.jpg")
        self.black_texture = Material("gfx/black.jpg")
        self.cube = Cube(
            position = [0,0,0],
            eulers = [0,0,0]
        )

        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy = 45, aspect = 1920/1080, 
            near = 0.1, far = 60, dtype=np.float32
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader,"projection"),
            1, GL_FALSE, projection_transform
        )
        self.modelMatrixLocation = glGetUniformLocation(self.shader,"model")
        
    def createShader(self, vertexFilepath, fragmentFilepath):

        with open(vertexFilepath,'r') as f:
            vertex_src = f.readlines()

        with open(fragmentFilepath,'r') as f:
            fragment_src = f.readlines()
        
        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                compileShader(fragment_src, GL_FRAGMENT_SHADER))
        
        return shader    
    def render(self):

        # generalapply =  pyrr.matrix44.multiply(
        #     m1=pyrr.matrix44.create_from_x_rotation(-math.pi/2),
        #     m2=pyrr.matrix44.create_from_z_rotation(-math.pi/2)
        # )
        generalapply =  pyrr.matrix44.create_identity(dtype=np.float32)
        model_transform_gen = pyrr.matrix44.multiply(
            m1=generalapply, 
            m2=pyrr.matrix44.create_look_at(
                np.array([30.0/2,20.0/2,40.0/2]),np.array([13,3,0]),np.array([0.0,0.0,1.0])                                                                #perspective
        )
        )

        
        #cube
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        """
            pitch: rotation around x axis
            roll:rotation around z axis
            yaw: rotation around y axis
        """
        #camera view setup
        
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform, 
            m2=model_transform_gen
        )

        
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform, 
            m2=pyrr.matrix44.create_from_translation(
                vec=np.array(self.cube.position),dtype=np.float32
            )
        )
        glUniformMatrix4fv(self.modelMatrixLocation,1,GL_FALSE,model_transform) #####LALALA
        self.wood_texture.use()
        self.cube_mesh = CubeMesh(self.cube_coordinate)
        glBindVertexArray(self.cube_mesh.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.cube_mesh.vertex_count)

     #shadow of cube
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        """
            pitch: rotation around x axis
            roll:rotation around z axis
            yaw: rotation around y axis
        """
        #camera view setup
        
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform, 
            m2=model_transform_gen
        )

        model_transform = pyrr.matrix44.multiply(
            m1=model_transform, 
            m2=pyrr.matrix44.create_from_eulers(
                eulers=np.radians(self.cube.eulers), dtype=np.float32
            )
        )
        
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform, 
            m2=pyrr.matrix44.create_from_translation(
                vec=np.array(self.cube.position),dtype=np.float32
            )
        )
        model_transform = pyrr.matrix44.multiply(
            m1=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,1]]),
            m2=model_transform
        )               
        glUniformMatrix4fv(self.modelMatrixLocation,1,GL_FALSE,model_transform) #####LALALA
        self.black_texture.use()
        self.cube_mesh = CubeMesh(self.cube_coordinate)
        glBindVertexArray(self.cube_mesh.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.cube_mesh.vertex_count)        
     #grid
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=model_transform_gen
        )
        
        #camera view setup
        
        # model_transform = pyrr.matrix44.multiply(
        #     m1=model_transform, 
        #     m2=pyrr.matrix44.create_look_at(
        #         np.array([5.0,5.0,5.0]),np.array([0.0,0.0,0.0]),np.array([0.0,0.0,1.0])
        #     )
        # )



        glUniformMatrix4fv(self.modelMatrixLocation,1,GL_FALSE,model_transform) #####LALALA
        self.grid_mesh = GridMesh(160)
        glBindVertexArray(self.grid_mesh.vao)
        glDrawArrays(GL_LINES, 0, self.grid_mesh.vertex_count)     
    def destroyshader(self):
        glDeleteProgram(self.shader)
class CubeMesh:
    def __init__(self,vertices):
        # x, y, z, s, t   
        self.vertices = vertices
        self.vertex_count = len(self.vertices)//5
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))  #Each number has four bytes 4*5 = 20 20 is the length of a row
                                                                                #last variable is offset. The position has zero offset, but the texture has 3*4 = 12 offsets
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
    
    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))

class GridMesh:
    def __init__(self, size):

        self.vertices = []

        for i in range(size):
            self.vertices.append(i-80)
            self.vertices.append(-size)
            self.vertices.append(0)
            self.vertices.append(i-80)
            self.vertices.append(size - 1)
            self.vertices.append(0)
        for j in range(size):
            self.vertices.append(-size)
            self.vertices.append(j-80)
            self.vertices.append(0)            
            self.vertices.append(size - 1)
            self.vertices.append(j-80)
            self.vertices.append(0)        
        self.vertex_count = len(self.vertices)//3
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        #position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))

class Material:

    
    def __init__(self, filepath):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        image = pg.image.load(filepath).convert()
        image_width,image_height = image.get_rect().size
        img_data = pg.image.tostring(image,'RGBA')
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.texture)

    def destroy(self):
        glDeleteTextures(1, (self.texture,))



myApp = App(w)
