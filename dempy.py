#Import modules
import numpy as np
import math
import heapq
from matplotlib import pyplot as plt
import copy
import subprocess as sp
from osgeo import gdal, ogr
# import geopandas as gpd
import glob
import os, sys, traceback
import statsmod
# import numpy.lib.format
# import PIL.Image
# import pandas as pd

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

    def create_demstruct(self, demfile):
        gdalobj = gdal.Open(demfile)
        ar = np.flipud(gdalobj.ReadAsArray().astype(np.float))
        rasterInfo = gdalobj.GetGeoTransform()

        self.grid = ar
        self.ncols = gdalobj.RasterXSize
        self.nrows = gdalobj.RasterYSize
        yulcorner = rasterInfo[3]
        self.cellsize = np.array(rasterInfo[1])
        self.xllcorner = rasterInfo[0] + self.cellsize/2
        self.yllcorner = yulcorner - (self.nrows-1)*self.cellsize - self.cellsize/2
        self.x = self.xllcorner + np.arange(self.ncols)*self.cellsize
        self.y = self.yllcorner + np.arange(self.nrows)*self.cellsize

    def create_demstruct_from_hdf(self, demfile, band):
        gdalobj = gdal.Open(demfile)
        subgdalobj = gdalobj.GetSubDatasets()[band][0]
        subgdalobj = gdal.Open(subgdalobj)
        ar = np.flipud(subgdalobj.ReadAsArray().astype(np.float))
        rasterInfo = subgdalobj.GetGeoTransform()

        self.grid = ar
        self.ncols = subgdalobj.RasterXSize
        self.nrows = subgdalobj.RasterYSize
        yulcorner = rasterInfo[3]
        self.cellsize = np.array(rasterInfo[1])
        self.xllcorner = rasterInfo[0] + self.cellsize / 2
        self.yllcorner = yulcorner - (self.nrows - 1) * self.cellsize - self.cellsize / 2
        self.x = self.xllcorner + np.arange(self.ncols) * self.cellsize
        self.y = self.yllcorner + np.arange(self.nrows) * self.cellsize

        return self
class large_dem:
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

    def create_demstruct(self, demfile):
        gdalobj = gdal.Open(demfile)
        # ar = np.flipud(gdalobj.ReadAsArray().astype(np.float))
        rasterInfo = gdalobj.GetGeoTransform()

        self.ncols = gdalobj.RasterXSize
        self.nrows = gdalobj.RasterYSize
        yulcorner = rasterInfo[3]
        self.cellsize = np.array(rasterInfo[1])
        self.xllcorner = rasterInfo[0] + self.cellsize/2
        self.yllcorner = yulcorner - (self.nrows-1)*self.cellsize - self.cellsize/2
        self.x = self.xllcorner + np.arange(self.ncols)*self.cellsize
        self.y = self.yllcorner + np.arange(self.nrows)*self.cellsize

        self.grid = np.memmap(demfile, dtype=np.uint16, shape=(self.ncols, self.nrows))

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

def burn_dem(demstruct, streamstruct, burndepth):
    #Simple stream burning routine that subtracts a scalar value (burndepth) from the DEM along pixels covered by
    #the streamlines found in streamstruct.
    # Also could use gdal_rasterize.
    dembrn = copy.deepcopy(demstruct)

    #Convert coordinates of stream file to subscripts
    i, j = np.nonzero(streamstruct.grid > 0)
    x, y = ij_2_xy(i, j, streamstruct)
    idem, jdem = xy_2_ij(x, y, demstruct)

    #Subract burndepth along channel pixels
    dembrn.grid[idem, jdem] -= burndepth

    return dembrn
def chan_slope_from_window(demfoldername, X, Y, dist):
    # Function to extract channel slope from a window of a DEM. The function iterates through all the DEMs in the folder and uses the one that X and Y are in. It then calculates slope over length dist up and downstream.
    # Assumes vertical unit is feet and horizontal unit is meters.
    # winsize = (dist/math.sqrt(2)) + 10 #Smallest radius that will allow a channel length of dist, plus a little bit.
    winsize = 1000

    X = X[0].tolist()
    Y = Y[0].tolist()
    if isinstance(X, list):
        X = X[0]
        Y = Y[0]

    if os.path.isfile(demfoldername[0:-6] + '\\temp.tif'):
        os.remove(demfoldername[0:-6] + '\\temp.tif')

    demfiles = glob.glob(demfoldername)
    flag = 0
    #First, extract appropriate DEM window
    for dem in demfiles:

        xmin, ymin, xmax, ymax = extract_raster_extent(dem)
        if (X > xmin) and (X < xmax) and (Y > ymin) and (Y < ymax):

            if os.path.isfile(demfoldername[0:-6] + '\\temp.tif'):
                os.remove(demfoldername[0:-6] + '\\temp.tif')

            select_dem_subwindow(dem, demfoldername[0:-6] + '\\temp.tif', X - winsize, Y + winsize, X + winsize, Y - winsize)
            demwin = DEM()
            demwin.create_demstruct(demfoldername[0:-6] + '\\temp.tif')
            # plot_dem(demwin, -9999)
            # plt.show()

            I, J = xy_2_ij(X, Y, demwin)
            if (demwin.grid[I, J] > -9999) and (I > 0) and (J > 0):
                os.remove(demfoldername[0:-6] + '\\temp.tif')
                fdem = priority_flood(demwin)
                fd, area = d8_flow(fdem)
                X, Y = snap_outlet_nearest(area, [X], [Y], 100)[:-1]

                # write_arc_ascii(demwin, r'D:\Gus_data\scratch_work\temp\rawdem.asc', -9999)
                # write_arc_ascii(fdem, r'D:\Gus_data\scratch_work\temp\fdem.asc', -9999)

                # plot_dem(fdem, -9999)
                # plt.plot(X, Y, 'r.')
                # plt.show()

                xt = np.arange(X[0]-dist/2, X[0]+dist/2)
                circ = np.sqrt(np.abs((dist/2)**2 - (xt - X[0])**2)) + Y[0]

                # idx, jdx = np.nonzero(demwin.grid >= 0)
                # maxz = np.max(demwin.grid[idx, jdx])
                # minz = np.min(demwin.grid[idx, jdx])
                it, jt = xy_2_ij(X, Y, demwin)
                zt = demwin.grid[it, jt]
                plt.figure()
                plot_dem(area, -9999)
                plt.figure()
                plot_dem(demwin, [zt-100, zt+100])
                plt.plot(X, Y, 'r.')
                plt.plot(xt, circ, 'r')
                plt.show(block=False)
                coords = plt.ginput(1, timeout=0)
                plt.close('all')

                xdown, ydown = search_down_flow_direction(fdem, fd, coords[0][0], coords[0][1])
                # xup, yup = search_up_flow_direction(fd, area, X, Y)

                # plt.figure()
                # plot_dem(area, -9999)
                # plt.plot(xdown, ydown, '.')
                # plt.plot(xup, yup, '.r')
                # plt.show()

                break

            else:
                continue
        else:
            continue

    inddown, Ldown = extract_evenly_spaced_points(xdown, ydown, dist)
    # indup, Lup = extract_evenly_spaced_points(xup, yup, 100)


    # Get slope by fitting line to elevation data
    ldown = np.cumsum(np.sqrt(np.diff(xdown[:inddown[0]])**2 + np.diff(ydown[:inddown[0]])**2))
    # lup = np.cumsum(np.sqrt(np.diff(xup[:indup[0]])**2 + np.diff(yup[:indup[0]])**2))

    id, jd = xy_2_ij(xdown[:inddown[0]], ydown[:inddown[0]], demwin)
    zd = demwin.grid[id, jd] * 0.3048 # convert vertical scale to meters
    zd = (zd[1:] + zd[:-1]) / 2
    # iu, ju = xy_2_ij(xup[:indup[0]], yup[:indup[0]], demwin)
    # zu = demwin.grid[iu, ju]
    # zu = (zu[1:] + zu[:-1]) / 2
    #
    # L = np.hstack((ldown[::-1], lup + np.max(ldown)))
    # Z = np.hstack((zd, zu))

    theta = statsmod.linear_regress_normal(ldown.reshape(ldown.shape[0], 1), zd.reshape(zd.shape[0], 1))

    S = np.abs(theta[1][0])
    print S
    # S = abs((zu - zd) / (ldown[0] + lup[0]))

    return S
def clip_feature(infile, outfile, clippingfile):
    cmd = ['ogr2ogr', '-clipsrc', clippingfile, outfile, infile]
    sp.call(cmd)
def clip_raster_2_shape(inras, inshape, outras):
    cmd1 = ['gdalwarp', '-cutline', inshape, '-crop_to_cutline', '-dstalpha', inras, outras[:-4]+'.vrt']
    sp.call(cmd1)
    cmd2 = ['gdal_translate', '-co', 'compress=LZW', outras[:-4]+'.vrt', outras]
    sp.call(cmd2)
def d8_flow(*args):
    # Calculates the D8 flow direction routing from a filled dem.
    # args allows you to input one or two  arguments. The first is the dem structure to be filled. The second is a
    # different raster structure to be accumulated in conjunction with the dem, for example precipitation data.

    demstruct = args[0]
    if len(args) > 1:
        otherstruct = args[1]

    fd = copy.deepcopy(demstruct)
    area = copy.deepcopy(demstruct)

    data = demstruct.grid

    cellsize = demstruct.cellsize
    demstruct = None

    fd.grid = np.zeros_like(data,dtype=int)
    area.grid = np.ones_like(data, dtype=int)

    indnan = data <= -9999
    fd.grid[indnan] = -9999
    it,jt = np.asarray(np.nonzero((fd.grid - -9999) <= 1e-5))
    fd.grid[it,jt] = -9999
    area.grid[it,jt] = -9999

    inds = data.argsort(axis=None)[::-1]

    for ind in inds:
        i,j = np.unravel_index(ind, data.shape)

        if (i == 0) | (i == fd.nrows-1) | (j == 0) | (j == fd.ncols-1) | (fd.grid[i,j] == -9999):
            continue

        else:

            neighborSubs = [[i-1, j-1], [i-1, j], [i-1, j+1], [i, j-1], [i, j+1], [i+1, j-1], [i+1, j], [i+1, j+1]]

            slopes = []
            for cell in neighborSubs:
                dx = fd.x[cell[1]] - fd.x[j]
                dy = fd.y[cell[0]] - fd.y[i]
                dL = np.sqrt(dx**2 + dy**2)
                slope = (data[i, j] - data[cell[0], cell[1]])/dL
                slopes.append(slope)

            maxind = np.argmax(slopes)

            if maxind == 0:
                fd.grid[i,j] = 8
            elif maxind == 1:
                fd.grid[i,j] = 4
            elif maxind == 2:
                fd.grid[i,j] = 2
            elif maxind == 3:
                fd.grid[i,j] = 16
            elif maxind == 4:
                fd.grid[i,j] = 1
            elif maxind == 5:
                fd.grid[i,j] = 32
            elif maxind == 6:
                fd.grid[i,j] = 64
            elif maxind == 7:
                fd.grid[i,j] = 128
            # if maxind == 0:
            #     fd.grid[i,j] = 32
            # elif maxind == 1:
            #     fd.grid[i,j] = 64
            # elif maxind == 2:
            #     fd.grid[i,j] = 128
            # elif maxind == 3:
            #     fd.grid[i,j] = 16
            # elif maxind == 4:
            #     fd.grid[i,j] = 1
            # elif maxind == 5:
            #     fd.grid[i,j] = 8
            # elif maxind == 6:
            #     fd.grid[i,j] = 4
            # elif maxind == 7:
            #     fd.grid[i,j] = 2
            else:
                print 'cell %r, %r is a ?' %(i,j)

            if area.grid[neighborSubs[maxind][0],neighborSubs[maxind][1]] > 0:
                area.grid[neighborSubs[maxind][0],neighborSubs[maxind][1]] += area.grid[i, j]
                if len(args) > 1:
                    otherstruct.grid[neighborSubs[maxind][0], neighborSubs[maxind][1]] += otherstruct.grid[i, j]

    area.grid = area.grid*(cellsize**2)
    if len(args) > 1:
        return fd, area, otherstruct
    else:
        return fd, area
def extract_evenly_spaced_points(X, Y, dist):
    i = 1
    ind = []
    cumuL = 0
    cumuLsave = []

    while i < len(X)-1:
        dx = X[i+1] - X[i]
        dy = Y[i+1] - Y[i]
        L = math.sqrt(dx**2 + dy**2)
        cumuL += L

        if cumuL >= dist:
            ind.append(i)
            cumuLsave.append(cumuL)
            cumuL = 0

        i += 1

    if len(ind) == 0:
        ind = [0]

    return ind, cumuLsave
def extract_grid_boundaries(orig_demstruct):
    demstruct = copy.deepcopy(orig_demstruct) #make a copy to operate on without modifying original dem
    # orig_demstruct = None
    # demstruct = orig_demstruct

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

    indnan = demstruct.grid <= -9999
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
def extract_pixel_value_at_points(rasterfile, X, Y):
    cmd = ['gdallocationinfo', '-geoloc', rasterfile, str(X), str(Y), '-valonly']
    return float(sp.check_output(cmd))
def extract_raster_extent(rasfile):
    gdalobj = gdal.Open(rasfile)
    geotransform = gdalobj.GetGeoTransform()

    ncols = gdalobj.RasterXSize
    nrows = gdalobj.RasterYSize
    cellsize = geotransform[1]

    xmin = geotransform[0] + cellsize/2
    xmax = xmin + cellsize*ncols

    ymax = geotransform[3] - cellsize/2
    ymin = ymax - nrows*cellsize

    return xmin, ymin, xmax, ymax
def hillshade(slopegrid,aspectgrid,azimuth,zenith):
    az = azimuth*math.pi/180
    ze = zenith*math.pi/180
    hill = (np.cos(slopegrid) * np.cos(math.pi/2 - ze)) + (np.sin(slopegrid) * np.sin(math.pi/2 - ze) * np.cos(az - aspectgrid))
    return hill
def ij_2_xy(i,j,gridstrct):
    x = gridstrct.xllcorner + np.asarray(j)*gridstrct.cellsize
    y = gridstrct.yllcorner + np.asarray(i)*gridstrct.cellsize
    # x = np.array(x)
    # y = np.array(y)
    return x, y
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
def long_profile(dem, fd, xo, yo, dist):
    x, y = search_down_flow_direction(fd, xo, yo)
    i, j = xy_2_ij(x, y, dem)
    z = dem.grid[i, j]
    z = (z[1:] + z[0:-1])/2

    dx = np.diff(x)
    dy = np.diff(y)
    L = np.cumsum(np.sqrt(dx**2 + dy++2))

    # Grab indices when distance is exceeded. Allows for mapping between long profile and map view
    idx = [0]
    for i in range(1, len(L)):
        if L[i] - L[idx[-1]] >= dist:
            idx.append(i)

    return z, L, x, y, idx
def mosaic_rasters(raslist, outpath):
    # cmd = ['gdal_merge.py', '-o', outpath]
    cmd = ['gdalwarp', '-r', 'bilinear', '-wm', '500']
    for path in raslist:
        cmd.append(path)

    cmd.append(outpath)
    sp.call(cmd)
    # cmd2 = ['gdal_translate', '-co', 'compress=LZW', outpath[:-4]+'.vrt', outpath]
    # sp.call(cmd2)
    # os.remove(outpath[:-4]+'.vrt')
def rasterize_vector_file(streamshapein, streamrasterout, cellsize, field2rasterize):

    #First, turn streamlines into raster; from GDAL/OGR cookbook
    # Define pixel_size and NoData value of new raster
    pixel_size = cellsize
    NoData_value = -9999

    # Filename of input OGR file
    vector_fn = streamshapein

    # Filename of the raster Tiff that will be created
    raster_fn = streamrasterout

    # Open the data source and read in the extent
    source_ds = ogr.Open(vector_fn)
    source_layer = source_ds.GetLayer()
    x_min, x_max, y_min, y_max = source_layer.GetExtent()

    # Create the destination data source
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)
    target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)

    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[0], options=['ATTRIBUTE=%s' % field2rasterize])
    target_ds = None
def plot_contour_map(demstruct,rng):
    if rng == -9999:
        vmin = np.min(demstruct.grid)
        vmax = np.max(demstruct.grid)
    else:
        vmin = rng[0]
        vmax = rng[1]

    X,Y = np.meshgrid(demstruct.x,demstruct.y)
    plt.contourf(X,Y,demstruct.grid, vmin=vmin, vmax=vmax)
    plt.axis('equal')
    plt.colorbar()
def plot_dem(demstruct,rng):
    demstruct.grid.astype(int)
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
    # plt.show()
def priority_flood(demstruct):
    #Priority flood pit-filling algorithm of Barnes et al., 2014, Computers and Geosciences

    filleddemstruct = copy.deepcopy(demstruct)
    # filleddemstruct = demstruct

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
                if (closed[ni[t],nj[t]] == 1) | (filleddemstruct.grid[ni[t], nj[t]] <= -9999):
                    continue
                else:
                    closed[ni[t],nj[t]] = 1

                    if filleddemstruct.grid[ni[t], nj[t]] <= filleddemstruct.grid[ci, cj]:
                        # incr = np.nextafter(filleddemstruct.grid[ni[t], nj[t]], filleddemstruct.grid[ci, cj])
                        # filleddemstruct.grid[ni[t], nj[t]] = np.nextafter(filleddemstruct.grid[ci, cj], filleddemstruct.grid[ci, cj] + 1)  # Add some increment
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

            if isinstance(ni, int):
                ni = [ni]
                nj = [nj]

            for t in range(0, len(ni)):
                if (closed[ni[t], nj[t]] == 1) | (filleddemstruct.grid[ni[t],nj[t]] <= -9999):
                    continue
                else:
                    closed[ni[t], nj[t]] = 1

                    if filleddemstruct.grid[ni[t], nj[t]] <= filleddemstruct.grid[ci, cj]:
                        # incr = np.nextafter(filleddemstruct.grid[ni[t], nj[t]], filleddemstruct.grid[ci, cj])
                        filleddemstruct.grid[ni[t], nj[t]] = filleddemstruct.grid[ci, cj] + 0.01  # Add some increment
                        count += 1
                        pit.append((ni[t], nj[t]))

                    else:
                        openq.qpush(filleddemstruct.grid[ni[t], nj[t]], (ni[t], nj[t]))


    return filleddemstruct
def project_raster_2utm(coordsys,zone,datum,file_in,file_out):
    zone = str(zone)
    cmd = ['gdalwarp', '-r', 'bilinear', '-t_srs', '+proj=%s +zone=%s +datum=%s' % (coordsys,zone,datum), file_in, file_out]
    sp.call(cmd)
def project_raster(file_in, file_out):
    # proj4 is a string giving the srs information. Can be found at spatialreference.org.
    proj4 = '+proj=lcc +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs' # Lambert conformal conic
    cmd = ['gdalwarp', '-t_srs', proj4, '-r', 'bilinear', file_in, file_out]
    sp.call(cmd)
def project_raster_with_compression(file_in, file_out):
    # proj4 is a string giving the srs information. Can be found at spatialreference.org.
    proj4 = '+proj=lcc +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs' # Lambert conformal conic
    cmd = ['gdalwarp', '-t_srs', proj4, '-r', 'bilinear', '-co', 'BIGTIFF=YES', '-of', 'vrt', file_in, file_out[:-4] + '.vrt']
    sp.call(cmd)
    cmd2 = ['gdal_translate', '-co', 'compress=LZW', '-co', 'BIGTIFF=YES', file_out[:-4] + '.vrt', file_out]
    sp.call(cmd2)
    os.remove(file_out[:-4] + '.vrt')
def project_hdf_band(prefix, filename, band, file_out):
    # proj4 is a string giving the srs information. Can be found at spatialreference.org.
    # a call to this function requires that the working directory is set to the location containing the files to be projected with os.chdir(dir).
    # find the band name using gdalinfo filename from the command line on one of the HDF files.
    proj4 = '+proj=lcc +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs'  # Lambert conformal conic
    cmd = ['gdalwarp', '-t_srs', proj4, '-r', 'bilinear', prefix + filename + band, file_out]
    sp.call(cmd)
def raster_polygonize(wd_path, inraster, outshapefile, band_num):
    # Copied from gdal/ogr cookbook.

    os.chdir(wd_path)

    # this allows GDAL to throw Python Exceptions
    gdal.UseExceptions()

    #  get raster datasource
    #
    src_ds = gdal.Open(inraster)
    if src_ds is None:
        print 'Unable to open %s' % inraster
        sys.exit(1)

    try:
        srcband = src_ds.GetRasterBand(band_num)
    except RuntimeError, e:
        # for example, try GetRasterBand(10)
        print 'Band ( %i ) not found' % band_num
        print e
        sys.exit(1)

    #  create output datasource
    #
    dst_layername = outshapefile
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_layername)
    dst_layer = dst_ds.CreateLayer(dst_layername, srs=None )

    gdal.Polygonize(srcband, None, dst_layer, -1, [], callback=None)
def raster_outline(inraster, outshapefile):
    cmd1 = ['gdalwarp', '-dstnodata', '-3.40282306074e+038', '-dstalpha', '-of', 'GTiff', inraster, 'foo1']
    cmd2 = ['gdal_polygonize.py', 'foo1', '-b', '2', '-f', "ESRI Shapefile", outshapefile]
    sp.call(cmd1)
    sp.call(cmd2)
    os.remove('foo1')
def resample_raster(file_in, file_out, mask_file, btif='no'):
    # mask_file is a raster to use to set the extent of file_out. The cells of file_out should match those of mask_fil
    gdalobj = gdal.Open(mask_file)
    rasterInfo = gdalobj.GetGeoTransform()
    cellsize = rasterInfo[1]
    xmin = rasterInfo[0]
    ymin = rasterInfo[3] - (gdalobj.RasterYSize - 1)*cellsize
    xmax = xmin + gdalobj.RasterXSize*cellsize
    ymax = ymin + gdalobj.RasterYSize*cellsize
    if (btif == 'yes') or (btif == 'y'):
        cmd = ['gdalwarp', '-tr', str(cellsize), str(cellsize), '-te', str(xmin), str(ymin), str(xmax), str(ymax),
            '-co', 'BIGTIFF=YES', '-of', 'vrt', '-r', 'bilinear', file_in[:-4] + '.vrt', file_out]
        sp.call(cmd)
        cmd2 = ['gdal_translate', '-co', 'compress=LZW', '-co', 'BIGTIFF=YES', file_out[:-4] + '.vrt', file_out]
        sp.call(cmd2)

    elif (btif == 'no') or (btif == 'n'):
        cmd = ['gdalwarp', '-tr', str(cellsize), str(cellsize), '-te', str(xmin), str(ymin), str(xmax), str(ymax),
            '-r', 'bilinear', file_in, file_out]
        sp.call(cmd)
    else:
        print 'bigtiff setting not understood'
def search_down_flow_direction(fd, xo, yo):
    # This function searches down the flow direction pathway using D8 flow routing.
    # It assumes the ArcInfo flow direction convention (base 2 starting at East, clockwise)

    i, j = xy_2_ij(xo, yo, fd)
    inew = int(i)
    jnew = int(j)
    numIters = 0

    while True:
        x, y = ij_2_xy(inew, jnew, fd)

        if inew <= 0 or jnew <= 0 or y > max(fd.y) or x > max(fd.x):
            i = i[0:-1]
            j = j[0:-1]
            break

        if fd.grid[inew, jnew] == 1:
            jnew += 1
        elif fd.grid[inew, jnew] == 2:
            jnew += 1
            inew -= 1
        elif fd.grid[inew, jnew] == 4:
            inew -= 1
        elif fd.grid[inew, jnew] == 8:
            inew -= 1
            jnew -= 1
        elif fd.grid[inew, jnew] == 16:
            jnew -= 1
        elif fd.grid[inew, jnew] == 32:
            jnew -= 1
            inew += 1
        elif fd.grid[inew, jnew] == 64:
            inew += 1
        elif fd.grid[inew, jnew] == 128:
            inew += 1
            jnew += 1
        else:
            break

        # print fd.grid[inew, jnew]
        # xn, yn = ij_2_xy(i, j, fd)
        # plot_dem(fd, -9999)
        # plt.plot(xn, yn ,'r.')
        # # plt.axis('equal')
        # plt.show()

        indall = np.ravel_multi_index((np.asarray(i), np.asarray(j)), fd.grid.shape)
        indthis = np.ravel_multi_index((np.asarray(inew), np.asarray(jnew)), fd.grid.shape)
        idx = np.nonzero(indthis == indall)
        indthis = np.delete(indthis, idx)
        inew, jnew = np.unravel_index(indthis, fd.grid.shape)

        # print str(dem.grid[inew, jnew]) + '\n'
        # xt, yt = ij_2_xy(i, j, fd)
        # plot_dem(fd, -9999)
        # plt.plot(xt, yt, 'r.')
        # plt.show()
        # print str(indall) + ', ' + str(indthis) + '\n'

        i = np.append(i, inew)
        j = np.append(j, jnew)
        numIters += 1

        # if numIters > 50000:
        #     print 'search_down_flow_direction() is caught in an infinite loop at i = %r and j = %r' % (itemp, jtemp)
        #     sys.exit()

    X,Y = ij_2_xy(i, j, fd)
    return X, Y
def search_up_flow_direction(fd, area, xo, yo):
    try:
        iout, jout = xy_2_ij(xo, yo, fd)
        itemp = copy.deepcopy(iout)
        jtemp = copy.deepcopy(jout)

        forever = 1
        flag = 0
        numIters = 0

        while forever == 1:
            i = np.arange(itemp-1, itemp+2)
            j = np.arange(jtemp-1, jtemp+2)
            i, j = np.meshgrid(i, j)
            i = i.reshape(1, 9)[0]
            j = j.reshape(1, 9)[0]

            #Find all cells that flow into current cell
            isave = []
            jsave = []
            c = 1
            for k in range(i.size):
                # for k2 in range(len(j)):
                inew = i[k]
                jnew = j[k]

                if (inew < 0) or (jnew < 0) or (inew > fd.grid.shape[0]) or (jnew > fd.grid.shape[1]):
                    continue

                if fd.grid[inew, jnew] == 1:
                    jnew += 1
                    if jnew == jtemp:
                        jsave.append(j[k])
                        isave.append(i[k])
                        c += 1
                elif fd.grid[inew, jnew] == 2:
                    jnew += 1
                    inew -= 1
                    if (jnew == jtemp) and (inew == itemp):
                        jsave.append(j[k])
                        isave.append(i[k])
                        c += 1
                elif fd.grid[inew, jnew] == 4:
                    inew -= 1
                    if inew == itemp:
                        isave.append(i[k])
                        jsave.append(j[k])
                        c += 1
                elif fd.grid[inew, jnew] == 8:
                    inew -= 1
                    jnew -= 1
                    if (jnew == jtemp) and (inew == itemp):
                        jsave.append(j[k])
                        isave.append(i[k])
                        c += 1
                elif fd.grid[inew, jnew] == 16:
                    jnew -= 1
                    if jnew == jtemp:
                        jsave.append(j[k])
                        isave.append(i[k])
                        c += 1
                elif fd.grid[inew, jnew] == 32:
                    jnew -= 1
                    inew += 1
                    if (jnew == jtemp) and (inew == itemp):
                        jsave.append(j[k])
                        isave.append(i[k])
                        c += 1
                elif fd.grid[inew, jnew] == 64:
                    inew += 1
                    if inew == itemp:
                        isave.append(i[k])
                        jsave.append(j[k])
                        c += 1
                elif fd.grid[inew, jnew] == 128:
                    inew += 1
                    jnew += 1
                    if (jnew == jtemp) and (inew == itemp):
                        jsave.append(j[k])
                        isave.append(i[k])
                        c += 1
                else:
                    flag = 1
                    break

            #test to see if included cells have already been processed
            # # idx = np.isin(np.vstack((isave, jsave)), np.vstack((iout, jout)))
            # idx = np.zeros_like(isave)
            # jdx = np.zeros_like(jsave)
            # for idxtemp in range(len(isave)):
            #     for jdxtemp in range(len(jsave)):
            #         if not (isave[idxtemp] in iout) and not (jsave[jdxtemp] in jout):
            #             idx[idxtemp] = 1
            #             jdx[jdxtemp] = 1
            # idx2 = [1 if (idx[iter] == 1) and (jdx[iter] == 1) else 0 for iter in range(len(idx))]
            # isave = np.asarray(isave)[np.asarray(idx2).astype(bool)]
            # jsave = np.asarray(jsave)[np.asarray(idx2).astype(bool)]

            if flag == 1:
                break

            if (itemp < 0) or (jtemp < 0) or (itemp >= fd.grid.shape[0]) or (jtemp >= fd.grid.shape[1]) or ((area.grid[itemp, jtemp])**2 <= area.cellsize**2+1): # I added the +1 to the last term after finding a DEM with a rounding error (?) in the cellsize, making it slightly larger than 1.0.
                break

            indall = np.ravel_multi_index((iout, jout), fd.grid.shape)
            indthis = np.ravel_multi_index((isave, jsave), fd.grid.shape)
            idx = np.nonzero(indthis == indall)
            indthis = np.delete(indthis, idx)
            isave, jsave = np.unravel_index(indthis, fd.grid.shape)


            #Next find cell of highest flow accumulation from all inflowing cells
            # ind = np.ravel_multi_index((isave, jsave), area.grid.shape)
            A = area.grid[isave, jsave]

            maxA = np.argmax(A)

            itemp, jtemp = isave[maxA], jsave[maxA]


            iout = np.append(iout, itemp)
            jout = np.append(jout, jtemp)
            numIters += 1

            if numIters > 50000:
                break
                # print 'search_up_flow_direction() is caught in an infinite loop at i = %r and j = %r' % (itemp, jtemp)
                # sys.exit()

            # print str(area.grid[itemp, jtemp]) + '\n'
            # if numIters % 10 == 0:
            #     xtemp, ytemp = ij_2_xy(iout, jout, fd)
            #     plot_dem(area, -9999)
            #     plt.plot(xtemp, ytemp, 'r.')
            #     plt.show()

        xout, yout = ij_2_xy(iout, jout, fd)
        if xout.size > 1:
            xout = xout[:-1]
            yout = yout[:-1]

    except Exception, e:
        traceback.print_exc(file=sys.stdout)

    return xout, yout
def select_dem_subwindow(filein, fileout, ulx, uly, lrx, lry):
    #for best results, the coordinates should be exact coordinates of the dem array. Otherwise there may be a pixel shift in the output dataset
    cmd = ['gdal_translate', '-projwin', str(ulx), str(uly), str(lrx), str(lry), filein, fileout]
    sp.call(cmd)
def snap_outlet_farthest(areastruct,xo,yo,radius):
    #Snaps a point to nearest cell of high flow accumulation, defined as the order of magnitude of the highest flow accumulation within a specified radius.
    # Radius is in number of cells

    X = []
    Y = []
    A = []

    it, jt = xy_2_ij(xo, yo, areastruct)

    for indit,i in enumerate(it):
            j = jt[indit]

            x,y = ij_2_xy(i,j,areastruct)

            while ((i - radius) < 0) or ((i + radius) >= areastruct.grid.shape[0]):
                radius -= 1

            while ((j - radius) < 0) or ((j + radius) >= areastruct.grid.shape[1]):
                radius -= 1

            iu = range(i-radius,i+radius,1)
            ju = range(j-radius,j+radius,1)

            # At = []
            # kind = []
            # k2ind = []
            # for k in range(len(iu)):
            #     for k2 in range(len(ju)):
            #         At.append(areastruct.grid[iu[k],ju[k2]])
            #         kind.append(k)
            #         k2ind.append(k2)

            # OM = math.log10(max(At))
            #
            # ind = [ind for ind in range(len(At)) if At[ind] >= 10**(OM-0.5)]
            #
            # iout = [iu[kind[ind[itemp]]] for itemp in range(len(ind))]
            # jout = [ju[k2ind[ind[jtemp]]] for jtemp in range(len(ind))]
            # xout, yout = ij_2_xy(iout, jout, areastruct)

            iu, ju = np.meshgrid(iu, ju)
            xout, yout = ij_2_xy(iu, ju, areastruct)

            # dx = [x - xout[ind2] for ind2 in range(len(xout))]
            # dy = [y - yout[ind2] for ind2 in range(len(yout))]
            # dL = [math.sqrt(dx[ind]**2 + dy[ind]**2) for ind in range(len(dx))]

            Avec = np.asarray(areastruct.grid[iu, ju])
            mindi, mindj = np.nonzero(Avec == np.max(Avec))
            # mind = Avec.index(max(Avec))


            X.append(xout[mindi, mindj])
            Y.append(yout[mindi, mindj])
            A.append((areastruct.cellsize**2) * areastruct.grid[iu[mindi, mindj],ju[mindi, mindj]])

    return X[0].tolist(), Y[0].tolist(), A[0].tolist()
def snap_outlet_nearest(areastruct,xo,yo,radius):
    #Snaps a point to nearest cell of high flow accumulation, defined as the order of magnitude of the highest flow accumulation within a specified radius.
    # Radius is in number of cells

    X = []
    Y = []
    A = []

    it, jt = xy_2_ij(xo, yo, areastruct)
    if isinstance(it, int):
        it = [it]
        jt = [jt]

    for indit,i in enumerate(it):
            j = jt[indit]

            x,y = ij_2_xy(i,j,areastruct)

            while ((i - radius) < 0) or ((i + radius) >= areastruct.grid.shape[0]):
                radius -= 1

            while ((j - radius) < 0) or ((j + radius) >= areastruct.grid.shape[1]):
                radius -= 1

            iu = range(i-radius,i+radius,1)
            ju = range(j-radius,j+radius,1)

            At = []
            kind = []
            k2ind = []
            for k in range(len(iu)):
                for k2 in range(len(ju)):
                    At.append(areastruct.grid[iu[k],ju[k2]])
                    kind.append(k)
                    k2ind.append(k2)

            OM = math.log10(max(At))

            ind = [ind for ind in range(len(At)) if At[ind] >= 10**(OM-0.5)]

            iout = [iu[kind[ind[itemp]]] for itemp in range(len(ind))]
            jout = [ju[k2ind[ind[jtemp]]] for jtemp in range(len(ind))]
            xout, yout = ij_2_xy(iout, jout, areastruct)

            dx = [x - xout[ind2] for ind2 in range(len(xout))]
            dy = [y - yout[ind2] for ind2 in range(len(yout))]
            dL = [math.sqrt(dx[ind]**2 + dy[ind]**2) for ind in range(len(dx))]

            mind = dL.index(min(dL))

            X.append(xout[mind])
            Y.append(yout[mind])
            A.append((areastruct.cellsize**2) * areastruct.grid[iout[mind],jout[mind]])

    return X, Y, A
def spatial_join(targ, source, jointype):
    #this function remains untested
    return gpd.sjoin(targ, source, op=jointype)
def sub2ind(siz, i,j):
    ncols = np.asarray(siz[0])
    ind = np.zeros_like(i)

    for k in range(0,i.shape[0]):
        ind[k] = j[k] + ncols*(i[k])

    return ind
def watershed(fd, xo, yo):

    #subscripts of outlet
    io, jo = xy_2_ij(xo, yo, fd)

    #initialize queues:
    closed = np.ones_like(fd.grid) * -9999
    openq = [(io, jo)]
    # openq = priority_queue()
    # openq.qpush(dem.grid[i, j], (i, j))

    while len(openq) > 0:
        # plot_dem(fd, -9999)
        # plt.imshow(closed)
        # plt.show()

        c = openq.pop()
        i = int(c[0])
        j = int(c[1])

        if closed[i, j] == 1:
            continue

        closed[i, j] = 1

        #Build D8 kernal
        ni = [i, i - 1, i - 1, i - 1, i, i + 1, i + 1, i + 1]
        nj = [j + 1, j + 1, j, j - 1, j - 1, j - 1, j, j + 1]

        #Find all neighboring cells that flow into current cell and place onto priority queue
        for k in range(len(ni)):
            inew = ni[k]
            jnew =nj[k]

            if (inew >= fd.grid.shape[0]) or (jnew >= fd.grid.shape[1]) or (inew < 0) or (jnew < 0):
                continue

            if (closed[inew, jnew] == 1):
                continue
            else:
                if fd.grid[inew, jnew] == 1:
                    jnew += 1
                    if (inew == i) and (jnew == j):
                        openq.append((ni[k], nj[k]))

                elif fd.grid[inew, jnew] == 2:
                    jnew += 1
                    inew -= 1
                    if (inew == i) and (jnew == j):
                        openq.append((ni[k], nj[k]))
                        # openq.qpush(dem.grid[inew, jnew], (inew, jnew))

                elif fd.grid[inew, jnew] == 4:
                    inew -= 1
                    if (inew == i) and (jnew == j):
                        openq.append((ni[k], nj[k]))
                        # openq.qpush(dem.grid[inew, jnew], (inew, jnew))

                elif fd.grid[inew, jnew] == 8:
                    jnew -= 1
                    inew -= 1
                    if (inew == i) and (jnew == j):
                        openq.append((ni[k], nj[k]))
                        # openq.qpush(dem.grid[inew, jnew], (inew, jnew))

                elif fd.grid[inew, jnew] == 16:
                    jnew -= 1
                    if (inew == i) and (jnew == j):
                        openq.append((ni[k], nj[k]))
                        # openq.qpush(dem.grid[inew, jnew], (inew, jnew))

                elif fd.grid[inew, jnew] == 32:
                    jnew -= 1
                    inew += 1
                    if (inew == i) and (jnew == j):
                        openq.append((ni[k], nj[k]))
                        # openq.qpush(dem.grid[inew, jnew], (inew, jnew))

                elif fd.grid[inew, jnew] == 64:
                    inew += 1
                    if (inew == i) and (jnew == j):
                        openq.append((ni[k], nj[k]))
                        # openq.qpush(dem.grid[inew, jnew], (inew, jnew))

                elif fd.grid[inew, jnew] == 128:
                    jnew += 1
                    inew += 1
                    if (inew == i) and (jnew == j):
                        openq.append((ni[k], nj[k]))
                        # openq.qpush(dem.grid[inew, jnew], (inew, jnew))


    #Get XY coordinates of watershed
    ind = np.nonzero(closed == 1)
    iout = ind[0]
    jout = ind[1]
    x, y = ij_2_xy(iout, jout, fd)

    #construct raster
    WS = DEM()
    WS.x = np.unique(x)
    WS.y = np.unique(y)
    WS.xllcorner = np.min(x)
    WS.yllcorner = np.min(y)
    WS.grid = np.ones((np.unique(y).shape[0], np.unique(x).shape[0]), dtype=int) * -3.40282306074e+038
    WS.cellsize = fd.cellsize
    WS.ncols = np.unique(x).shape[0]
    WS.nrows = np.unique(y).shape[0]
    i, j = xy_2_ij(x,y, WS)
    WS.grid[i, j] = 1

    return WS, iout, jout
def write_arc_ascii(struct, outfilePath, noDataValue):
    #modified from Sam Johnstone

    header = "ncols     %s\n" % struct.ncols
    header += "nrows    %s\n" % struct.nrows
    header += "xllcenter %s\n" % struct.xllcorner
    header += "yllcenter %s\n" % struct.yllcorner
    header += "cellsize %s\n" % struct.cellsize
    header += "NODATA_value %s" % noDataValue
    npArrayData = np.flipud(struct.grid)

    # np.savetxt(outfilePath, npArrayData, header=header, fmt=formatString, comments='')
    np.savetxt(outfilePath, npArrayData, header=header, comments='')
def write_geotiff(demstruct, outfilename, geotiff_ref):
    # geotiff_ref is a path to a geotif in the desired projection.
    [cols, rows] = np.flipud(demstruct.grid).shape
    driver = gdal.GetDriverByName('GTiff')
    outdata = driver.Create(outfilename, rows, cols, 1, gdal.GDT_Float32)

    gdalobj = gdal.Open(geotiff_ref)
    # geotrans = gdalobj.GetGeoTransform()
    proj = gdalobj.GetProjection()

    geotrans = []
    geotrans.append(np.min(demstruct.x) - demstruct.cellsize/2)
    geotrans.append(demstruct.cellsize)
    geotrans.append(0)
    geotrans.append(np.max(demstruct.y) - demstruct.cellsize/2)
    geotrans.append(0)
    geotrans.append(-demstruct.cellsize)

    outdata.SetGeoTransform(geotrans)
    outdata.SetProjection(proj)
    outdata.GetRasterBand(1).WriteArray(np.flipud(demstruct.grid))
    outdata.FlushCache()
    outdata = None
def xy_2_ij(xo,yo,gridstrct):
    dy = np.asarray(yo) - gridstrct.yllcorner
    dx = np.asarray(xo) - gridstrct.xllcorner
    i = np.round(dy / gridstrct.cellsize)
    j = np.round(dx / gridstrct.cellsize)
    i = np.array(i.astype(int))
    j = np.array(j.astype(int))
    return i, j
