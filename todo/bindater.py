# -*- coding: utf-8 -*-
"""
Gives the last date for data used in each bin

@author: Simon Streicher
"""

import pandas as pd
import numpy as np

from ranking.data_processing import split_tsdata

datapath = 'C:/faultmap_data/cases/test/data/testcase.csv'

outputpath = 'boxdates.csv'

# Need to read the actual input dates
datasource = pd.read_csv(open(datapath, 'rb'))

dates = datasource['Time']

boxsize = 3600
boxnum = 100
samplerate = 5

date_indexes = range(len(dates))

date_boxes = split_tsdata(date_indexes, samplerate, boxsize, boxnum)

end_dates_df = pd.DataFrame()

end_dates = []
for date_box in date_boxes:
    end_dates.append(dates[date_box[-1]])

end_dates_df['End dates'] = end_dates
#end_dates_df['End dates'] = pd.to_datetime(end_dates_df['End dates'],
#                                           dayfirst=True)
end_dates_df.to_csv(outputpath, index=False)
