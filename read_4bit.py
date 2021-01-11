import numpy as np
#from matplotlib import pyplot as plt
import time
def unpack_1bit(arr):
    tmpr=np.bitwise_and(np.right_shift(arr,7),1)
    tmpi=np.bitwise_and(np.right_shift(arr,6),1)
    tmp1=tmpr+1J*tmpi

    tmpr=np.bitwise_and(np.right_shift(arr,5),1)
    tmpi=np.bitwise_and(np.right_shift(arr,4),1)
    tmp2=tmpr+1J*tmpi

    tmpr=np.bitwise_and(np.right_shift(arr,3),1)
    tmpi=np.bitwise_and(np.right_shift(arr,2),1)
    tmp3=tmpr+1J*tmpi

    tmpr=np.bitwise_and(np.right_shift(arr,1),1)
    tmpi=np.bitwise_and(arr,1)
    tmp4=tmpr+1J*tmpi


    dat=np.zeros([tmp1.shape[0],4*tmp1.shape[1]],dtype='complex')
    dat[:,0::4]=tmp1
    dat[:,1::4]=tmp2
    dat[:,2::4]=tmp3
    dat[:,3::4]=tmp4

    return 2*dat-1-1J
def unpack_4bit(arr):
    t1=time.time()
    tmpr=np.asarray(np.bitwise_and(np.right_shift(arr,4),15),dtype='int8')
    tmpi=np.asarray(np.bitwise_and(arr,15),dtype='int8')
    t2=time.time()
    tmpr[tmpr>7]=tmpr[tmpr>7]-16
    tmpi[tmpi>7]=tmpi[tmpi>7]-16
    t3=time.time()
    print('time to arrayify is '+repr(t2-t1) + ' and to zero-center is ' + repr(t3-t2))
    return np.asarray(tmpr+1J*tmpi,dtype='complex64')
    
    #return tmpr,tmpi

#f=open('1b1471-1971.raw','r')
#f=open('1b494-985.raw','r')
def read_4bit_new(fname):
    t1=time.time()
    f=open(fname,'r')

    #crud=np.fromfile(f,'>Q',40)
    #for i in range(len(crud)):
    #    print(i,'   ',crud[i])

    header_bytes=np.fromfile(f,'>Q',1)[0]
    packet_size=np.fromfile(f,'>Q',1)[0]
    spectra_per_packet=np.fromfile(f,'>Q',1)[0]
    nbit=np.fromfile(f,'>Q',1)[0]
    have_trimble=np.fromfile(f,'>Q',1)[0]
    gps_week=np.fromfile(f,'>Q',1)[0]
    gps_seconds=np.fromfile(f,'>Q',1)[0]
    #nchan=np.fromfile(f,'>Q',1)[0]
    nchan=np.int(header_bytes/8-10)
    chans=np.fromfile(f,'>Q',nchan)
    print('chans are ' + repr(chans) + ' with nbit ' + repr(nbit))
    lat=np.fromfile(f,">d",1)[0]
    lon=np.fromfile(f,">d",1)[0]
    el=np.fromfile(f,">d",1)[0]
    print("lat/lon are : ",repr(lat) + "  "  + repr(lon)+ "  " + repr(el))

    ts=np.fromfile(f,'c')
    f.close()
    t2=time.time()
    print('read raw data in ',t2-t1)
    nchan=nchan*2 #because we have two inputs

    npacket=np.uint64(len(ts)/packet_size)
    ts_raw=ts[:npacket*packet_size]
    ts=np.reshape(ts_raw,[npacket,packet_size])
    tmp=ts[:,:4]
    
    tmp=np.reshape(tmp,tmp.size)
    specno=np.fromstring(tmp,'>i')

    #print(specno[:5])
    #print(specno[-5:])
    print("diff mean is ",np.mean(np.diff(specno)))
    ts=np.reshape(ts[:,4:],ts.shape[0]*(ts.shape[1]-4))
    ts=np.reshape(ts,[np.uint64(len(ts)/nchan),nchan])
    ts=np.fromstring(ts,dtype='uint8')
    ts=np.reshape(ts,[np.uint64(len(ts)/nchan),nchan])

    t3=time.time()

    dat=unpack_4bit(ts)
    dat1=dat[:,::2]
    dat2=dat[:,1::2]

    print('Time to read is '+ repr(t2-t1) + ' and time to process is ' + repr(t3-t2))
    del(dat)
    return dat1,dat2

def read_4bit(fname):

    f=open(fname,'r')
    t1=time.time()
    header_size=np.fromfile(f,'>Q',1)[0]
    packet_size=np.fromfile(f,'>Q',1)[0]
    nchan=np.fromfile(f,'>Q',1)[0]
    print('have ' + repr(nchan) + ' channels.')
    specs_per_packet=np.fromfile(f,'>Q',1)[0]
    have_trimble=np.fromfile(f,'>Q',1)[0]
    aa=np.fromfile(f,'>Q',1)[0]
    chans=np.fromfile(f,'>Q',nchan)
    print("starting/stopping channels are ",chans[0],chans[-1],chans)
    gps_time=np.fromfile(f,'>Q',2)
    print("time is ",gps_time)
    pos=np.fromfile(f,'>d',3)
    print("position is ",pos)
#aa=np.fromfile(f,'>i',1)[0]
#nb=np.uint64(packet_size-4)
#print("type is ",type(nb))
#crap=np.fromfile(f,'c',nb)
#bb=np.fromfile(f,'>i',1)[0]
#print("next entry is ",aa,bb)
    ts=np.fromfile(f,'c')
    f.close()
    t2=time.time()
    print('read raw data in ',t2-t1)

    npacket=np.uint64(len(ts)/packet_size)
    ts_raw=ts[:npacket*packet_size]
    ts=np.reshape(ts_raw,[npacket,packet_size])
    tmp=ts[:,:4]
    
    tmp=np.reshape(tmp,tmp.size)
    specno=np.fromstring(tmp,'>i')
    #print(specno[:5])
    #print(specno[-5:])
    print("diff mean is ",np.mean(np.diff(specno)))
    ts=np.reshape(ts[:,4:],ts.shape[0]*(ts.shape[1]-4))
    ts=np.reshape(ts,[np.uint64(len(ts)/nchan),nchan])
    ts=np.fromstring(ts,dtype='uint8')
    ts=np.reshape(ts,[np.uint64(len(ts)/nchan),nchan])

    t3=time.time()

    dat=unpack_4bit(ts)
    dat1=dat[:,::2]
    dat2=dat[:,1::2]

    print('Time to read is '+ repr(t2-t1) + ' and time to process is ' + repr(t3-t2))
    del(dat)
    return dat1,dat2

def read_header(fname):

    with open(fname,'r') as f:
        header_bytes=np.fromfile(f,'>Q',1)[0]
        nchan=np.int(header_bytes/8-10)
        header = {
            'packet_size':np.fromfile(f,'>Q',1)[0],
            'spectra_per_packet':np.fromfile(f,'>Q',1)[0],
            'nbit':np.fromfile(f,'>Q',1)[0],
            'have_trimble':np.fromfile(f,'>Q',1)[0],
            'gps_week':np.fromfile(f,'>Q',1)[0],
            'gps_seconds':np.fromfile(f,'>Q',1)[0],
            'nchan':np.int(header_bytes/8-10), 
            'chans':np.fromfile(f,'>Q',nchan),
            'lat':np.fromfile(f,">d",1)[0],
            'lon':np.fromfile(f,">d",1)[0],
            'el':np.fromfile(f,">d",1)[0]
            }
    return header


#mycorr=np.mean(dat1*np.conj(dat2),axis=0)
#auto1=np.mean(dat1*np.conj(dat1),axis=0)
#auto2=np.mean(dat2*np.conj(dat2),axis=0)

#mycorr_mat=np.dot(dat2.transpose(),np.conj(dat2))
#mycorr=np.diag(mycorr_mat)/(2.0*dat2.shape[0]) #bonus 2.0 is so a perfectly correlated signal will be one.

#plt.ion()
#plt.clf()
##plt.plot(np.angle(mycorr),'*')
#plt.plot(np.real(mycorr))
#plt.plot(np.imag(mycorr))
#plt.legend(["Real xcorr","Imag xcorr"])
#plt.savefig("4bit_xcorr.png")
