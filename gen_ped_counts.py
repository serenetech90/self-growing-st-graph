import pandas as pd
import numpy as np
import glob

def main():
    allfiles = '/home/siri0005/Documents/self-growing-spatial-graph/self-growing-gru-offline_avgPool/data/crowds_attrb/*.csv'
    f = sorted(glob.glob(allfiles))
    pd.options.display.max_rows = 20000
    for i in {0}:#1,2,3,4,5,6,7
        data = np.genfromtxt(f[i],delimiter=',')#)
        dframe = pd.DataFrame({'fid': data[:,0], 'pid': data[:,1] , 'x': data[:,2]\
                      , 'y': data[:,3]})
        counts = dframe.groupby('fid')['pid'].count()
        wf = open(
            '/home/siri0005/Documents/self-growing-spatial-graph/self-growing-gru-offline_avgPool/data/crowds_attrb/ped_counts_{0}.txt'.format(i),
            'w')
        wf.write(str(counts)+'\n')
        wf.write(str(max(counts))+'\n')

    wf.close()

    return


if __name__ == '__main__':
    main()