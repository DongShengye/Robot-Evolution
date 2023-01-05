#Import Library

import numpy as np
import concurrent.futures
import math
import matplotlib.pyplot as plt
import time 
import copy
import random
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired

#subprocess--------------------------------------------------------------

def subprocess(abcd):
    cube_coordinate = []
    
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

    arena = []
    springs = []
    masses = []
    avoid_collide = np.zeros([50,50,50]) #important
    cubes = []
    position_foropengl = []
    #constants
    k = 10000
    g = 10 #m/s^2
    m = 60 #kg
    a = 1  #global definition
    w = 40  #global definition
    T = 0
    dt = 0.0005
    r_p = 90000

    rest = 1
    cube_indices = []
    cubey = []
    
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
        
        for i in range(0,4):     #4 tpyes of materials
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
            
            a = 1
            qwe = 0
            
            for w in range(len(springs_new)):
                
                if set(springs_new[w].idfor2).issubset(weird_buffer):
                
                    springs_new[w].a = a
                    springs_new[w].b = b
                    springs_new[w].c = c
                    springs_new[w].k = k
    
    def App(metacube,gene):
        



        Us = 0.5
        Uk = 0.2
        indi1 = []
        indi2 = []
        buffer11 = []
        p_subset = []
        cube_coordinate = []
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
            if t >= 4*math.pi/w:
                running = False 
            t = t+dt
            

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

                else:
                    
                    FN = sumed_forces[2]-r_p*masses[i].indices[2]-m*g
                    FH = math.sqrt(sumed_forces[0]**2+sumed_forces[1]**2)
                    
                    if FH<-FN*Us:
                        Ftotal = sumed_forces+np.array([0,0,-m*g])+np.array([0,0,-r_p*masses[i].indices[2]]) 

                    if FH>=-FN*Us:
                        denominator = math.sqrt(masses[i].velocity[0]**2+masses[i].velocity[1]**2)

                        Ftotal = sumed_forces-np.array([masses[i].velocity[0]/denominator*(Uk*FN),masses[i].velocity[1]/denominator*(Uk*FN),0])+np.array([0,0,-m*g])+np.array([0,0,-r_p*masses[i].indices[2]])#-damping*masses[i].velocity

                Ftotal_store.append(Ftotal)
            Fitness = 0 

            for i in range(len(masses)):
                masses[i].acceleration = Ftotal_store[i]/m
                
                masses[i].velocity = masses[i].velocity+masses[i].acceleration*dt
                masses[i].indices = masses[i].indices+masses[i].velocity*dt
                if running == False:
                    Fitness += masses[i].indices[0]
        buffer11.append((Fitness-Original_fitness)/len(masses)/(4*math.pi/w))            
        for i in range(len(gene)):
            buffer11.append(gene[i])
        for i in range(len(metacube)):
            buffer11.append(metacube[i])          
            

        return buffer11
    
    metacube = meta_grow([3,3,0],12)
    
    create_range = cubemaker(metacube, a)
    springs_new = springkiller(springs)
    backup_springs = copy.deepcopy(springs_new)
    backup_masses = copy.deepcopy(masses)
    
    
    springs_new = copy.deepcopy(backup_springs)
    masses = copy.deepcopy(backup_masses)
    gene = gene_creation(create_range)
    #cubey = np.array(cubey)
    
    spring_assginer(cubey,gene)
    gene_sub = App(metacube,gene)
    
    
    return gene_sub

def subprocess1(gene_massdistri):

    
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

    arena = []
    springs = []
    masses = []
    avoid_collide = np.zeros([50,50,50]) #important
    cubes = []
    position_foropengl = []
    #constants
    k = 10000
    g = 10 #m/s^2
    m = 60 #kg
    a = 1  #global definition
    w = 40  #global definition
    T = 0
    dt = 0.0005
    r_p = 90000

    rest = 1
    cube_indices = []
    cubey = []

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
    def spring_assginer(cubey,gene):
        for i in range(len(cubey)):
            
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
            
            a = 1
            qwe = 0
            
            for w in range(len(springs_new)):
                
                if set(springs_new[w].idfor2).issubset(weird_buffer):
                
                    springs_new[w].a = a
                    springs_new[w].b = b
                    springs_new[w].c = c
                    springs_new[w].k = k
    
    def App(gene,metacube):
        



        Us = 0.5
        Uk = 0.2
        indi1 = []
        indi2 = []
        buffer11 = []
        p_subset = []
        cube_coordinate = []
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
            if t >= 4*math.pi/w:
                running = False 
            t = t+dt
            

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

                else:
                    
                    FN = sumed_forces[2]-r_p*masses[i].indices[2]-m*g
                    FH = math.sqrt(sumed_forces[0]**2+sumed_forces[1]**2)
                    
                    if FH<-FN*Us:
                        Ftotal = sumed_forces+np.array([0,0,-m*g])+np.array([0,0,-r_p*masses[i].indices[2]]) 

                    if FH>=-FN*Us:
                        denominator = math.sqrt(masses[i].velocity[0]**2+masses[i].velocity[1]**2)

                        Ftotal = sumed_forces-np.array([masses[i].velocity[0]/denominator*(Uk*FN),masses[i].velocity[1]/denominator*(Uk*FN),0])+np.array([0,0,-m*g])+np.array([0,0,-r_p*masses[i].indices[2]])#-damping*masses[i].velocity

                Ftotal_store.append(Ftotal)
            Fitness = 0 

            for i in range(len(masses)):
                masses[i].acceleration = Ftotal_store[i]/m
                
                masses[i].velocity = masses[i].velocity+masses[i].acceleration*dt
                masses[i].indices = masses[i].indices+masses[i].velocity*dt
                if running == False:
                    Fitness += masses[i].indices[0]
        buffer11.append((Fitness-Original_fitness)/len(masses)/(4*math.pi/w))            
        for i in range(len(gene)):
            buffer11.append(gene[i])
        for i in range(len(metacube)):
            buffer11.append(metacube[i])  
        
        return buffer11
    metacube = gene_massdistri[1+4:]
    
    create_range = cubemaker(metacube, a)
    springs_new = springkiller(springs)
    backup_springs = copy.deepcopy(springs_new)
    backup_masses = copy.deepcopy(masses)   
    springs_new = copy.deepcopy(backup_springs)
    masses = copy.deepcopy(backup_masses)

    #cubey = np.array(cubey)
    gene_purematerial = gene_massdistri[1:5]
    
    spring_assginer(cubey,gene_purematerial)
    gene_sub = App(gene_purematerial,metacube)
    return gene_sub

if __name__ == '__main__':
    arena = []
    # file_log = open("data45.txt", "a")
    # file_log2 = open("data56.txt", "a")
    
    
    with ProcessPool() as pool:
        future = pool.map(subprocess, range(50), timeout=20)                                     # 50 generations 
    
        iterator = future.result()

        while True:
            try:
                result = next(iterator)
                arena.append(result)
            except StopIteration:
                break
            except TimeoutError as error:
                print("function took longer than %d seconds" % error.args[1])
 
#q = subprocess()
    #subprocess()
    print("original 50 over")
    saver = []
    diversity_plot = []

    #iteration start
    for qw in range(5000): 
        sub_diversity_plot = []
        print("iteration",qw,"starts")  
        st = time.time()
        buffer1 = []


        for ert in range(8):                                                             # 8 mutations
            godchoice = random.randint(0,len(arena)-1)
            picker = random.randint(1,len(arena[0])-1)
            
            if picker <=4:
                picker_material = random.randint(0,3)
                if picker_material == 0:
                    portion = random.uniform(-3,3)
                    arena[godchoice][picker][picker_material] = portion*copy.deepcopy(arena[godchoice][picker][picker_material])
                else:
                    portion = random.uniform(-1.05,1.05)
                    arena[godchoice][picker][picker_material] = portion*copy.deepcopy(arena[godchoice][picker][picker_material])                
            else:
                picker_xyz = random.randint(0,2)

                originalxyz = copy.deepcopy(arena[godchoice][picker])
                #print(arena[godchoice][picker][picker_xyz])
                readytoadd = copy.deepcopy(arena[godchoice][picker][picker_xyz])+random.choice([-1,1])
                originalxyz[picker_xyz] = readytoadd
                flag = 0
                for rt in range(len(arena[godchoice])-5):
                    #print('originalxyz',originalxyz)
                    #print('arena[godchoice][rt]',arena[godchoice][rt+5])
                    if originalxyz == arena[godchoice][rt+5]:
                        flag = 1
                        
                if flag == 0 and originalxyz[2]>=0:
                    arena[godchoice][picker] = originalxyz
                    
        print("mutation 8 over")  

        with ProcessPool() as pool:
            future = pool.map(subprocess, range(20), timeout=20)                                   #20 generations
        
            iterator = future.result()

            while True:
                try:
                    result = next(iterator)
                    arena.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print("function took longer than %d seconds" % error.args[1]) 
        
        print("supply 20 over")

        child_gene = []  
        supp = 100-len(arena)
        print("how many crossover needed for maintain",supp)
        for i in range(supp):                                                               #  crossovers
            parent1 = random.randint(0,len(arena)-1)
            flag = 1
            while flag ==1:
                parent2 = random.randint(0,len(arena)-1)
                if parent1 != parent2:
                    flag = 0
            cutpoint1 = random.randint(2,int((len(arena[0])-1)/2))
            cutpoint2 = random.randint(int((len(arena[0])-1)/2),int((len(arena[0])-1)))
            child_gene.append(arena[parent1][:cutpoint1]+arena[parent2][cutpoint1:cutpoint2]+arena[parent1][cutpoint2:])
        #print(child_gene)
        with ProcessPool() as pool:
            future = pool.map(subprocess1, child_gene, timeout=20)
    
            iterator = future.result()

            while True:
                try:
                    result = next(iterator)
                    arena.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print("function took longer than %d seconds" % error.args[1])
        print("crossover",supp,"over")
        #Mutation - module
              
        #print(arena)
        arena.sort(reverse=True,key=lambda arena: arena[0])
        for div in range(len(arena)):
            sub_diversity_plot.append(arena[div][0])
        diversity_plot.append(sub_diversity_plot)

        arena = arena[:int(len(arena)/2)]
        saver.append(arena[0][0])
        print(len(arena))
        
        print("gen ",qw,"answer: ",'\n',arena[0],'\n',arena[1],'\n',arena[2],'\n')
        et = time.time()
        print("gen ",qw,"time",et-st)
        if qw%100 == 0:

            print('\n')
            print("For best results",saver)
            print('\n')
            print("For diversity plot",diversity_plot)
            print('\n')


            
