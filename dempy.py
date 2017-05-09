#Import modules
import numpy as np
import math
import heapq
from matplotlib import pyplot as plt
import copy
import subprocess as sp

class DemStruct:
    # Define the class to hold DEM structure including gridded data and spatial information.
    def __init__(self):
        self.grid = np.empty(0)
        self.ncols = np.empty(0)
        self.nrows = np.empty(0)
        self.xllcorner = np.empty(0)
        self.yllcorner = np.empty(0)
        self.cellsize = np.empty(0)
        self.x = np.empty(0)
        self.y = np.empty(0)


class DEM:
    # Define the class to hold DEM structure including gridded data and spatial information.
    def __init__(self):
        self.grid = np.empty(0)
        self.ncols = np.empty(0)
        self.nrows = np.empty(0)
        self.xllcorner = np.empty(0)
        self.yllcorner = np.empty(0)
        self.cellsize = np.empty(0)
        self.x = np.empty(0)
        self.y = np.empty(0)

    def create_demstruct(self, gdalobj):
        ar = np.flipud(gdalobj.ReadAsArray().astype(np.float))
        rasterInfo = gdalobj.GetGeoTransform()

        self.grid = ar
        self.ncols = gdalobj.RasterXSize
        self.nrows = gdalobj.RasterYSize
        self.xllcorner = rasterInfo[0]
        yulcorner = rasterInfo[3]
        self.cellsize = np.array(rasterInfo[1])
        self.yllcorner = yulcorner - (self.nrows-1)*self.cellsize
        self.x = self.xllcorner + np.arange(self.ncols)*self.cellsize
        self.y = self.yllcorner + np.arange(self.nrows)*self.cellsize

        return self


class priority_queue:
    #Establish a priority queue that contains the items to be flooded and the priority of the items

    def __init__(self):
        self.__qitems = [] #The queue itself. Each item is a tuple containing row,column subscripts into the dem array
        self.__qcounter = 0 #the secondary priority
        self.__qnumitems = 0 #the number of items in the queue
        heapq.heapify(self.__qitems)

    def qpop(self):
        priority, counter, item = heapq.heappop(self.__qitems)
        self.__qnumitems -=1
        return priority, item

    def qpush(self, priority, item):
        self.__qcounter += 1
        self.__qnumitems += 1
        entry = [priority, self.__qcounter, item]
        heapq.heappush(self.__qitems,entry)

    def qisfull(self):
        return self.__qnumitems > 0


def xy_2_ij(xo,yo,gridstrct):
    dy = yo - gridstrct.yllcorner
    dx = xo - gridstrct.xllcorner
    i = round(dy / gridstrct.cellsize)
    j = round(dx / gridstrct.cellsize)
    i = np.array(i)
    j = np.array(j)
    return int(i), int(j)


def ij_2_xy(i,j,gridstrct):
    x = gridstrct.xllcorner + j*gridstrct.cellsize
    y = gridstrct.yllcorner + i*gridstrct.cellsize
    x = np.array(x)
    y = np.array(y)
    return x, y


def sub2ind(siz, i,j):
    ncols = np.asarray(siz[0])
    ind = np.zeros_like(i)

    for k in range(0,i.shape[0]):
        ind[k] = j[k] + ncols*(i[k])

    return ind

def ind2sub(siz,ind):
    ncols = np.asarray(siz[0])
    i = np.zeros_like(ind)
    j = np.zeros_like(ind)

    for k in range(0,ind.shape[0]):
        i[k] = ind[k]/ncols
        ind = ind.astype('float')
        ncols = ncols.astype('float')
        j[k] = ind[k]%ncols
    return i, j


def search_down_flow_direction(fd,xo,yo):
    #This function searches down the flow direction pathway using D8 flow routing.
    #It assumes the ArcInfo flow direction convention (base 2 starting at East, clockwise)
    i,j = xy_2_ij(xo,yo,fd)
    inew = int(i)
    jnew = int(j)

    while True:
        x,y = ij_2_xy(inew,jnew,fd)

        if inew < 0 or jnew < 0 or y > max(fd.y) or x > max(fd.x):
            i = i[0:-1]
            j = j[0:-1]
            break

        if fd.grid[inew,jnew] == 1:
            jnew += 1
        elif fd.grid[inew,jnew] == 2:
            jnew += 1
            inew -= 1
        elif fd.grid[inew,jnew] == 4:
            inew -= 1
        elif fd.grid[inew,jnew] == 8:
            inew -= 1
            jnew -= 1
        elif fd.grid[inew,jnew] == 16:
            jnew -= 1
        elif fd.grid[inew,jnew] == 32:
            jnew -= 1
            inew += 1
        elif fd.grid[inew,jnew] == 64:
            inew += 1
        elif fd.grid[inew,jnew] == 128:
            inew += 1
            jnew += 1
        else:
            break

        i = np.append(i, inew)
        j = np.append(j, jnew)

    X,Y = ij_2_xy(i,j,fd)
    return X, Y


def hillshade(slopegrid,aspectgrid,azimuth,zenith):
    az = azimuth*math.pi/180
    ze = zenith*math.pi/180
    hill = (np.cos(slopegrid) * np.cos(math.pi/2 - ze)) + (np.sin(slopegrid) * np.sin(math.pi/2 - ze) * np.cos(az - aspectgrid))
    return hill


def extract_grid_boundaries(orig_demstruct):
    demstruct = copy.deepcopy(orig_demstruct) #make a copy to operate on without modifying original dem
    #Window DEM with -9999
    siz = demstruct.grid.shape
    nancol = -9999*np.ones((siz[0],1),dtype='float')
    demstruct.grid = np.append(nancol, demstruct.grid, axis=1)
    demstruct.grid = np.append(demstruct.grid, nancol, axis=1)

    siz = demstruct.grid.shape
    nanrow = -9999*np.ones((1,siz[1]),dtype='float')
    demstruct.grid = np.append(nanrow, demstruct.grid, axis=0)
    demstruct.grid = np.append(demstruct.grid, nanrow, axis=0)

    #Adjust struct to incorporate grid size changed
    demstruct.xllcorner = demstruct.xllcorner - demstruct.cellsize
    demstruct.yllcorner = demstruct.yllcorner - demstruct.cellsize
    demstruct.x = np.append(demstruct.x[0]-demstruct.cellsize, demstruct.x)
    demstruct.x = np.append(demstruct.x, demstruct.x[-1]+demstruct.cellsize)
    demstruct.y = np.append(demstruct.y[0]-demstruct.cellsize, demstruct.y)
    demstruct.y = np.append(demstruct.y, demstruct.y[-1]+demstruct.cellsize)
    demstruct.nrows = demstruct.nrows + 2
    demstruct.ncols = demstruct.ncols + 2

    siz = demstruct.grid.shape

    indnan = demstruct.grid < 0
    demstruct.grid[indnan] = -9999

    i,j = np.asarray(np.nonzero(demstruct.grid == -9999))

    isave = []
    jsave = []

    for c in range(0,i.shape[0]):
        #ic,jc = ind2sub(demstruct.grid.shape, np.array([c]))
        ic = i[c]
        jc = j[c]
        iker = np.asarray([ic, ic+1, ic+1, ic+1, ic, ic-1, ic-1, ic-1])
        jker = np.asarray([jc+1, jc+1, jc, jc-1, jc-1, jc-1, jc, jc+1])
        rm = (iker>=0) & (jker>=0) & (iker<siz[0]) & (jker<siz[1])
        iker = iker[rm]
        jker = jker[rm]
        #ndt = sub2ind(demstruct.grid.shape,iker,jker)

        #Search cells around indx(c)  in D8 sense
        for c2 in range(0,iker.shape[0]):
            #test = demstruct.grid[indt[c2]]
            test = demstruct.grid[iker[c2],jker[c2]]
            if test != -9999:
                #ind = np.append(ind,indt[c2])
                isave = np.append(isave,iker[c2])
                jsave = np.append(jsave,jker[c2])

    isave = isave - 1 #account for added border NaNs
    jsave = jsave - 1
    return np.asarray(isave,dtype='int'), np.asarray(jsave,dtype='int')


def priority_flood(demstruct):
    #Priority flood pit-filling algorithm of Barnes et al., 2014, Computers and Geosciences

    filleddemstruct = demstruct

    #Extract and sort boundary cells into initial priority queue
    i, j = extract_grid_boundaries(filleddemstruct)

    elevs = filleddemstruct.grid[i,j]

    #initialize queues
    closed = np.zeros_like(filleddemstruct.grid, dtype=bool)

    openq = priority_queue() #establish an instance of a priority queue to place cells that have not yet been modified
    for c in range(0,i.shape[0]):
        row,col = i[c], j[c]
        closed[row,col] = 1
        openq.qpush(filleddemstruct.grid[row,col],(row,col))

    pit = [] #establish pit as a plain queue; pit will contain tuples of row, col subscripts but priority is based solely on in/out order

    siz = filleddemstruct.grid.shape

    #Begin priority flood algorithm
    count = 0
    while openq.qisfull() | len(pit) > 0: #run algorithm as long as open and pit contain items that need to be filled

        if len(pit) > 0:
            c = pit.pop(0) #remove and return first element from pit
            ci = c[0]
            cj = c[1]

            #closed[ci,cj] = 1

            #find subscripts of neighboring cells
            ni = [ci, ci-1, ci-1, ci-1, ci, ci+1, ci+1, ci+1]
            nj = [cj+1, cj+1, cj, cj-1, cj-1, cj-1, cj, cj+1]
            idx = [idx for idx in range(0,len(ni)) if (ni[idx] >= 0) & (nj[idx] >= 0) & (ni[idx] < siz[0]) & (nj[idx] < siz[1])]
            ni = [ni[ind] for ind in idx]
            nj = [nj[ind] for ind in idx]

            if isinstance(ni,int):
                ni = [ni]
                nj = [nj]

            for t in range(0,len(ni)):
                if (closed[ni[t],nj[t]] == 1) | (filleddemstruct.grid[ni[t],nj[t]] < 0):
                    continue
                else:
                    closed[ni[t],nj[t]] = 1

                    if filleddemstruct.grid[ni[t], nj[t]] <= filleddemstruct.grid[ci, cj]:
                        filleddemstruct.grid[ni[t], nj[t]] = filleddemstruct.grid[ci, cj] + 0.01  # Add some increment
                        count += 1
                        pit.append((ni[t], nj[t]))

                    else:
                        openq.qpush(filleddemstruct.grid[ni[t], nj[t]], (ni[t], nj[t]))

        else:
            c = openq.qpop()
            c = c[-1]
            ci = c[0]
            cj = c[1]

            #closed[ci,cj] = 1

            #find subscripts of neighboring cells
            ni = [ci, ci-1, ci-1, ci-1, ci, ci+1, ci+1, ci+1]
            nj = [cj+1, cj+1, cj, cj-1, cj-1, cj-1, cj, cj+1]

            idx = [idx for idx in range(0,len(ni)) if (ni[idx] >= 0) & (nj[idx] >= 0) & (ni[idx] < siz[0]) & (nj[idx] < siz[1])]
            ni = [ni[ind] for ind in idx]
            nj = [nj[ind] for ind in idx]

            if isinstance(ni,int):
                ni = [ni]
                nj = [nj]

            for t in range(0,len(ni)):
                if (closed[ni[t],nj[t]] == 1) | (filleddemstruct.grid[ni[t],nj[t]] < 0):
                    continue
                else:
                    closed[ni[t],nj[t]] = 1

                    if filleddemstruct.grid[ni[t],nj[t]] <= filleddemstruct.grid[ci,cj]:
                        filleddemstruct.grid[ni[t],nj[t]] = filleddemstruct.grid[ci,cj] + 0.01 #Add some some increment
                        count += 1
                        pit.append((ni[t],nj[t]))

                    else:
                        openq.qpush(filleddemstruct.grid[ni[t],nj[t]],(ni[t],nj[t]))


    return filleddemstruct


def d8_flow(demstruct):
    #Calculates the D8 flow direction codes from a filled dem.

    data = demstruct.grid
    fd = np.zeros_like(data,dtype=int)
    area = np.ones_like(data, dtype=int)

    indnan = demstruct.grid < 0
    demstruct.grid[indnan] = -9999
    it,jt = np.asarray(np.nonzero(demstruct.grid == -9999))
    fd[it,jt] = -9999
    area[it,jt] = -9999

    inds = data.argsort(axis=None)[::-1]

    for ind in inds:
        i,j = np.unravel_index(ind,data.shape)

        if (i == 0) | (i == demstruct.nrows-1) | (j == 0) | (j == demstruct.ncols-1) | (fd[i,j] == -9999):
            continue

        else:

            neighborSubs = [[i-1,j-1], [i-1,j], [i-1,j+1], [i,j-1], [i,j+1], [i+1,j-1], [i+1,j], [i+1,j+1]]

            slopes = []
            for cell in neighborSubs:
                if cell[0] >= demstruct.y.shape[0]:
                    sp = 1

                dx = demstruct.x[cell[1]] - demstruct.x[j]
                dy = demstruct.y[cell[0]] - demstruct.y[i]
                dL = np.sqrt(dx**2 + dy**2)
                slope = (data[i,j] - data[cell[0],cell[1]])/dL
                slopes.append(slope)

            maxind = np.argmax(slopes)

            if maxind == 0:
                fd[i,j] = 32
            elif maxind == 1:
                fd[i,j] = 64
            elif maxind == 2:
                fd[i,j] = 128
            elif maxind == 3:
                fd[i,j] = 16
            elif maxind == 4:
                fd[i,j] = 1
            elif maxind == 5:
                fd[i,j] = 8
            elif maxind == 6:
                fd[i,j] = 4
            elif maxind == 7:
                fd[i,j] = 2
            else:
                print 'cell %r, %r is a ?' %(i,j)

            if area[neighborSubs[maxind][0],neighborSubs[maxind][1]] > 0:
                area[neighborSubs[maxind][0],neighborSubs[maxind][1]] += area[i, j]

    return fd, area*(demstruct.cellsize**2)


def plot_dem(demstruct,rng):
    xmin = np.min(demstruct.x)
    xmax = np.max(demstruct.x)
    ymin = np.min(demstruct.y)
    ymax = np.max(demstruct.y)
    extent = (xmin, xmax, ymin, ymax)

    if rng == -9999:
        vmin = np.min(demstruct.grid)
        vmax = np.max(demstruct.grid)
    else:
        vmin = rng[0]
        vmax = rng[1]

    plt.imshow(demstruct.grid, vmin=vmin, vmax=vmax, origin='lower', extent=extent)
    plt.axis('equal')
    plt.colorbar()
    plt.show()


def plot_contour_map(demstruct,rng):
    if rng == -9999:
        vmin = np.min(demstruct.grid)
        vmax = np.max(demstruct.grid)
    else:
        vmin = rng[0]
        vmax = rng[1]

    X,Y = np.meshgrid(demstruct.x,demstruct.y)
    plt.contourf(X,Y,demstruct.grid)
    plt.axis('equal')
    plt.colorbar()
    plt.show()


def write_arc_ascii(struct, outfilePath, noDataValue, formatString):
    #modified from Sam Johnstone

    header = "ncols     %s\n" % struct.ncols
    header += "nrows    %s\n" % struct.nrows
    header += "xllcenter %s\n" % struct.xllcorner
    header += "yllcenter %s\n" % struct.yllcorner
    header += "cellsize %s\n" % struct.cellsize
    header += "NODATA_value %s" % noDataValue
    npArrayData = np.flipud(struct.grid)

    np.savetxt(outfilePath, npArrayData, header=header, fmt=formatString, comments='')


def project_raster(coordsys,zone,datum,file_in,file_out):
    zone = str(zone)
    cmd = ['gdalwarp', '-t_srs', '+proj=%s +zone=%s +datum=%s' % (coordsys,zone,datum), file_in, file_out]
    sp.call(cmd)