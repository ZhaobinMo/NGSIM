# coding: utf-8

from pyproj import Proj
import pyproj

'''
Convert the cordanate
out: dfSet
'''
# EPSG 4326(WGS84): the common lat/lon system
# EPSG 2229: California State Plane Coordinate System, Zone 5, 
    #NAD83 http://spatialreference.org/ref/epsg/nad83-california-zone-5-ftus/ ( but use http://epsg.io/4326 here)
    #bOUND:5528230.8160, 1384701.5952, 7751890.9134, 2503239.6463 (epsg2229)
        #-121.3600, 32.7500, -114.1300, 35.8100(WGS84)
        

wgs84 = Proj("+proj=longlat +datum=WGS84 +no_defs ") # there should not be " " between the "init" and the "EPSG
epsg2229 = Proj("+proj=lcc +lat_1=35.46666666666667 +lat_2=34.03333333333333 +lat_0=33.5 +lon_0=-118 +x_0=2000000.0001016+y_0=500000.0001016001 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=us-ft +no_defs", preserve_units=True)  
             
                # Pyproj expects degrees (lon, lat) or meters (x,y) as units but the unit of Projection: 2263 isUNIT["US survey foot"...

#print( pyproj.transform(epsg2229,wgs84, 6451128.359, 1873301.311) )

class CoordinateConverter(object):
    def __init__(self, df, colXin, colYin, colXout, colYout, projIn, projOut):
        self.df = df
        self.colXin = colXin
        self.colYin = colYin
        self.colXout = colXout
        self.colYout = colYout
        self.projIn = projIn
        self.projOut = projOut
    def convert(self, xy):  
        xIn = xy[0:xy.index(' ')]; yIn = xy[xy.index(' ')+1 : ]
        xOut, yOut = pyproj.transform(self.projIn, self.projOut, xIn, yIn)
        xy_out = str(xOut) + ' ' + str(yOut)
        return xy_out
    def returnX(self, xyStr):
        x = xyStr[ 0  :xyStr.index(' ')]
        return x
    def returnY(self, xyStr):
        y = xyStr[xyStr.index(' ') + 1 : ]
        return y
    def convertCordinate(self):
        col_x = self.colXin; col_y = self.colYin
        col_x_out = self.colXout; col_y_out = self.colYout
        df = self.df
        # assemble into one column for applying the 'apply'
        df[col_x] = df[col_x].astype(str); df[col_y] = df[col_y].astype(str)
        df['xy'] = df[col_x] + ' ' + df[col_y]
        # begin convert
        df['xy_out'] = df['xy'].apply(self.convert)
        # return the X and Y from the assemble OUT
        df[col_x_out] = df['xy_out'].apply(self.returnX)
        df[col_y_out] = df['xy_out'].apply(self.returnY)
        # delete the temp
        del(df['xy'], df['xy_out'])
        return df