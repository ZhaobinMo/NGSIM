# coding: utf-8

import pandas as pd
from sodapy import Socrata
import datetime
from dateutil import tz
import numpy as np



from CoordinateConverter import CoordinateConverter, wgs84, epsg2229
class NgsimExtractor(object):
    def __init__(self, location, gap):
        print('begin init')
        self.location = location
        self.gap = gap
        if location == "\"us-101\"":
            self.minGlobalTime = 1118846979700
        elif location == "\"i-80\"":
            self.minGlobalTime = 1113433136100
        elif location == "\"peachtree\"":
            self.minGlobalTime = 1163019100000
        else:
            print('the locaiton is wrong')
        print('end init')
        print('gap={}'.format(gap))
        print('location = {}'.format(location))
    def extractData( self ):
        ''''''
        # extract the raw date
        client = Socrata("data.transportation.gov", None)
        LIMIT = 1000000
        results = client.get("jqsx-yj2r", where = "global_time %{}=0 and location = {} ".format( self.gap, self.location) , limit = LIMIT)
        results_df = pd.DataFrame.from_records(results)
        
        # warning about the LIMIT
        print('datasize={}  limit={}'.format(len(results_df), LIMIT))
        if LIMIT == len(results_df):
            print('Warning: the LIMIT may be small, set it in the NgsimExtractor')
        
        # fix the problem about the peachtree (global time lacks three '0' digit at the end)
        if self.location == "\"peachtree\"":
            results_df['date_time'] = (results_df['global_time'] + '000').apply(self.epoch2datetime)
        else:
            results_df['date_time'] = results_df['global_time'].apply(self.epoch2datetime)
        
        # CONVERT: convert the coordinate for plot consideration
        CVTer = CoordinateConverter(results_df, 'global_x', 'global_y', 'wgs_x', 'wgs_y', epsg2229, wgs84)
        dfSet = CVTer.convertCordinate()
        dfSet['longitude'] = dfSet['wgs_x'].astype(float);  dfSet['latitude'] = dfSet['wgs_y'].astype(float)        
        
        # CONVERT: some data type convert
        dfSet['vehicle_id'] = dfSet['vehicle_id'].astype(int)
        dfSet['v_class'] = dfSet['v_class'].astype(int)
        dfSet['lane_id'] = dfSet['lane_id'].astype(int)
        
        dfSet['v_vel'] = dfSet['v_vel'].astype(float)
        dfSet['v_vel'] = dfSet['v_vel'] * 0.681818 # convert feet/s to mph
        
        dfSet['local_y'] = dfSet['local_y'].astype(float)
        dfSet['local_y'] = dfSet['local_y'] * 0.3048 # convert feet to meter
        
        # NEW COLUMN: t_diff
        dfSet['t_diff'] = (dfSet['global_time'].astype(np.int64) - self.minGlobalTime)/1000; dfSet['t_diff'] = dfSet['t_diff'].astype(int)
        
        # NEW COLUMN: unit_id
        dfSet = self.addUnitId(dfSet)
        
        # CUT: use the columns that we want
        dfSet = dfSet[[ 'unit_id', 'vehicle_id', 'global_time', 'date_time', 't_diff' , 'latitude', 'longitude', 
                       'local_y','v_vel','v_class','lane_id','frame_id','total_frames', 'location']]
        
        # CONVERT: global time to local time
        dfSet = dfSet.sort_values(['unit_id', 'global_time'])
        dfSet = self.global2localTime(dfSet)
        
        return dfSet
        
    def addUnitId(self, df):
        # mask to find the not-continuous point
        df = df.sort_values([ 'vehicle_id', 'global_time', 'v_class'])
        mask_vehicleidNOTequal = df['vehicle_id'].ne(df['vehicle_id'].shift(1))
        mask_vclassNOTequal = df['v_class'].ne(df['v_class'].shift(1))
        mask_timeNOTcontinuous = (df['global_time'].astype(np.int64) ).ne( (df['global_time'].shift(1).fillna(0).astype(np.int64) + self.gap ) )

        # add the 'unit_id' to the not-continuous point
        mask = mask_vehicleidNOTequal | mask_vclassNOTequal | mask_timeNOTcontinuous
        df['unit_id_pre'] = 0
        df['unit_id'] = 0
        df.loc[mask, 'unit_id_pre'] = 1
        df['unit_id'] = df['unit_id_pre'].cumsum()
        
        #
        del( df['unit_id_pre'] )
        return df


    def epoch2datetime(self, epochTime):
        # Epoch in dataset is 10^13 (millsec)
        # however, it is 10^10 (sec) in the python package, so there should be a '.' before the last 3 digit 
        dateTime = datetime.datetime.utcfromtimestamp(float(str(epochTime)[:-3]+'.'+str(epochTime)[-3:])).strftime('%Y-%m-%d %H:%M:%S')
        return dateTime

    def global2localTime(self, df):
        from_zone = tz.gettz('UTC')
        to_zone = tz.gettz('America/Los_Angeles')
        df['date_time'] = df['global_time'].apply(self.epoch2datetime)
        df['date_time'] = pd.to_datetime(df['date_time'])
        df['date_time'] = df['date_time'].apply(lambda x: x.replace(tzinfo = from_zone).astimezone(to_zone).strftime('%Y-%m-%d %H:%M:%S'))
        return df