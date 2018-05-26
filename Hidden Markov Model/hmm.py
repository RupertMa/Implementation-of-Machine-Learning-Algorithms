import numpy as np
import itertools

def loadDataset():
    with open('hmm-data.txt') as f:
        grid=[]
        for line in f.readlines():
            line=line.rstrip()
            if 'Grid' in line:
                grid=[]
                switch='Grid'
                continue
            if 'Location' in line:
                location={}
                switch='Location'
                continue
            if 'Noisy' in line:
                noisy=[]
                switch='Noisy'
                continue
            if switch=='Grid':
                if line!='':
                    grid.append(line.split(' '))
                    continue   
            if switch=='Location':
                if line!='':
                    key=line.split(': ')[0]
                    values=list(map(float,line.split(': ')[1].split(' ')))
                    location[key]=values
                    continue
            if switch=='Noisy':
                if line!='':
                    noisy.append(list(map(float,line.split())))
                    continue
        return np.array(grid),location,noisy

def free_cells_index(grid):
    free_loc=[]
    it = np.nditer(grid, flags=['multi_index'])
    while not it.finished:
        if it[0]=='1':
            free_loc.append(it.multi_index)
        it.iternext()
    return free_loc
    
def distance_to_towers(grid,location):
    L2=lambda x,y:((x[0]-y[0])**2+(x[1]-y[1])**2)**0.5
    distance_matrix=[]
    free_loc=free_cells_index(grid)
    for cell in free_loc:
        temp=[]
        for coordinates in location.values():
            dist=L2(cell,coordinates)
            temp.append([0.7*dist,1.3*dist])
        distance_matrix.append(temp)
    return distance_matrix

def possible_states(noisy,distance_matrix,free_loc):
    states=[]
    for step in noisy:
        temp=[]
        for index,cell in enumerate(distance_matrix):
            if step[0]<cell[0][0] or step[0]>cell[0][1]:continue
            if step[1]<cell[1][0] or step[1]>cell[1][1]:continue
            if step[2]<cell[2][0] or step[2]>cell[2][1]:continue
            if step[3]<cell[3][0] or step[3]>cell[3][1]:continue
            temp.append(free_loc[index])
        states.append(temp)
    return states

def find_neighbors(free_loc,position):
    neighbors=[]
    if position[0]-1<0:x=[position[0]+1]
    elif position[0]+1>9:x=[position[0]-1]
    else:x=[position[0]-1,position[0]+1]
    if position[1]-1<0:y=[position[1]+1]
    elif position[1]-1>9:y=[position[1]-1]
    else:y=[position[1]-1,position[1]+1]
    neighbors.extend(list(itertools.product(x, [position[1]])))
    neighbors.extend(list(itertools.product([position[0]],y)))
    for neighbor in neighbors:
        if neighbor not in free_loc:
            neighbors.remove(neighbor)
    return neighbors


def Viterbi(noisy,free_loc,start_p,trans_p,emis_p):
    T1=np.zeros((len(free_loc),len(noisy)))
    T2=np.zeros((len(free_loc),len(noisy)))
    for index,state in enumerate(free_loc):
        T1[index,0]=start_p[state]*emis_p[state][0] 
    for i in range(1,len(noisy)):
        for j,state in enumerate(free_loc):
            temp=[]
            for k in range(len(free_loc)):
                temp.append(T1[k,i-1]*trans_p[k,j]*emis_p[state][i])
            #if sum(temp)>0:
                #print(temp)
            #temp=np.array(temp)
            T1[j,i]=max(temp)
            T2[j,i]=np.argmax(temp)
        #print(T1[:,i])
    Path=[]
    zt=np.argmax(T1[:,10])
    xT=free_loc[zt]
    Path.append(xT)
    zt_1=zt
    for i in reversed(range(1,11)):
        zt_1=int(T2[zt_1,i])
        xT_1=free_loc[zt_1]
        Path.append(xT_1)
    return list(reversed(Path))

def Viterbi_sum_logarithm(noisy,free_loc,start_p,trans_p,emis_p):
    T1=np.zeros((len(free_loc),len(noisy)))
    T2=np.zeros((len(free_loc),len(noisy)))
    for index,state in enumerate(free_loc):
        num=start_p[state]*emis_p[state][0]
        if num==0:
            T1[index,0]=num
        else:
            T1[index,0]=np.abs(np.log(start_p[state])+np.log(emis_p[state][0]))   #sum logarithms 
    for i in range(1,len(noisy)):
        for j,state in enumerate(free_loc):
            temp=[]
            for k in range(len(free_loc)):
                num=T1[k,i-1]*trans_p[k,j]*emis_p[state][i]
                if num==0:
                    temp.append(num)
                else:
                    temp.append(T1[k,i-1]+np.abs(np.log(trans_p[k,j]))+np.abs(np.log(emis_p[state][i])))
            if sum(temp)==0:
                T1[j,i]=max(temp)
                T2[j,i]=np.argmax(temp)
            else:
                minimum=np.inf
                for index,num in enumerate(temp):
                    if num!=0 and num<minimum:
                        minimum=num
                        minind=index
                T1[j,i]=minimum
                T2[j,i]=minind
    Path=[]
    minimum=np.inf
    for index,num in enumerate(T1[:,10]):
        if num!=0 and num<minimum:
            minimum=num
            minind=index
    zt=minind
    xT=free_loc[zt]
    Path.append(xT)
    zt_1=zt
    for i in reversed(range(1,11)):
        zt_1=int(T2[zt_1,i])
        xT_1=free_loc[zt_1]
        Path.append(xT_1)
    return list(reversed(Path))


def main():
    grid,location,noisy=loadDataset()
    free_loc=free_cells_index(grid)
    distance_matrix=distance_to_towers(grid,location)
    states=possible_states(noisy,distance_matrix,free_loc)
    start_p={}
    trans_p=np.zeros((len(free_loc),len(free_loc)))
    #calculate start probability and transition probability
    for index,cell in enumerate(free_loc):
        start_p[cell]=1./len(free_loc)
        neighbors=find_neighbors(free_loc,cell)
        for neighbor in neighbors:
            trans_p[index,free_loc.index(neighbor)]=1./len(neighbors)
    #calculate emission probability
    emis_p={}
    for cell in free_loc:
        temp={}
        for step in range(11):
            temp[step]=0.
        emis_p[cell]=temp
    for index,step in enumerate(states):
        for cell in step:
            temp=1.
            for dist_tower in distance_matrix[free_loc.index(cell)]:
                temp=temp*(1./(np.floor(dist_tower[1]*10)-np.ceil(dist_tower[0]*10)+1))
            emis_p[cell][index]=temp 
    Path=Viterbi_sum_logarithm(noisy,free_loc,start_p,trans_p,emis_p)
    #Path=Viterbi(noisy,free_loc,start_p,trans_p,emis_p)
    print('The most likely trajectory is: ',Path)

if __name__=="__main__":
    main()





