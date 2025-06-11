import numpy as np
import math

def HVM_read(f_path, nx, ny, nz, ns, Bx0,By0,Bz0,lleb, llion, llentr,llpress,outformat):
    print("Reading in ", f_path)

    a0 = 0.0
    a1 = 0
    a2 = 0
    a3 = 0

    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    b3 = 0.0
    b4 = 0.0
    b5 = 0.0

    time = []
    dimt = 0
    filename = f_path + "tempo.dat"

    with open(filename, 'r') as file:
        for line in file:
            a0 = float(line.strip())
            time.append(a0)
            dimt += 1

    print("TIMES TO BE READ ARE:")
    ic =0
    for it in range(dimt):
        print(time[it])
        ic=ic+1


    dim1 = nx
    dim2 = ny
    dim3 = nz
    
    b0 = np.zeros((dim1, dim2, dim3))
    b1 = np.zeros((dim1, dim2, dim3))
    b2 = np.zeros((dim1, dim2, dim3))
    b3 = np.zeros((dim1, dim2, dim3))
    b4 = np.zeros((dim1, dim2, dim3))
    b5 = np.zeros((dim1, dim2, dim3))

    ns = ic # int(input('Number of times saved in tempo.dat '))
    print('')

    llion = (llion == 'y')
    if llion:
        filename = f_path + 'Ion.bin'
        time = []
        px = []
        py = []
        pz = []
        rho = []
        if outformat == 0:
            with open(filename, 'rb') as file:
                file.read(4)
                for i in range(ns):
                    time.append(np.fromfile(file,dtype = "f8",count = 1, offset  = 0) )

                    nx,ny,nz = np.fromfile(file, dtype = "i4",count = 3, offset = 0)

                    TOTAL_SIZE = nx*ny*nz
                    SHAPE = (nx,ny,nz)
                    file.read(8)
                    px.append(np.fromfile(file, dtype = "f8",count = TOTAL_SIZE, offset = 0).reshape(SHAPE,order = "F") )
                    py.append(np.fromfile(file, dtype = "f8",count = TOTAL_SIZE, offset = 0).reshape(SHAPE,order = "F") )
                    pz.append(np.fromfile(file, dtype = "f8",count = TOTAL_SIZE, offset = 0).reshape(SHAPE,order = "F") )
                    rho.append(np.fromfile(file, dtype = "f8",count = TOTAL_SIZE, offset = 0).reshape(SHAPE,order = "F") )
                    file.read(8)

    rho = np.rollaxis(np.asarray(rho),0,4)
    den = 1.0 + rho 

    px = np.rollaxis(np.asarray(px),0,4)
    py = np.rollaxis(np.asarray(py),0,4)
    pz = np.rollaxis(np.asarray(pz),0,4)

    uxp = px / (rho + 1.0)
    uyp = py / (rho + 1.0)
    uzp = pz / (rho + 1.0)


    lleb = (lleb == 'y')
    if lleb:
        filename = f_path + 'EB.bin'
        time = []
        ex = []
        ey = []
        ez = []
        bx = []
        by = []
        bz = []
        if outformat == 0:
            with open(filename, 'rb') as file:
                file.read(4)
                for i in range(ns):
                    time.append(np.fromfile(file,dtype = "f8",count = 1,offset  = 0))

                    nx,ny,nz = np.fromfile(file,dtype = "i4",count = 3,offset = 0)

                    TOTAL_SIZE = nx*ny*nz
                    SHAPE = (nx,ny,nz)
                    file.read(8)
                    ex.append(np.fromfile(file,dtype = "f8",count = TOTAL_SIZE,offset = 0).reshape(SHAPE,order = "F"))
                    ey.append(np.fromfile(file,dtype = "f8",count = TOTAL_SIZE,offset = 0).reshape(SHAPE,order = "F"))
                    ez.append(np.fromfile(file,dtype = "f8",count = TOTAL_SIZE,offset = 0).reshape(SHAPE,order = "F"))
                    bx.append(np.fromfile(file,dtype = "f8",count = TOTAL_SIZE,offset = 0).reshape(SHAPE,order = "F"))
                    by.append(np.fromfile(file,dtype = "f8",count = TOTAL_SIZE,offset = 0).reshape(SHAPE,order = "F"))
                    bz.append(np.fromfile(file,dtype = "f8",count = TOTAL_SIZE,offset = 0).reshape(SHAPE,order = "F"))
                    file.read(8)

    bx = np.rollaxis(np.asarray(bx),0,4) + Bx0
    by = np.rollaxis(np.asarray(by),0,4) + By0
    bz = np.rollaxis(np.asarray(bz),0,4) + Bz0
    ex = np.rollaxis(np.asarray(ex),0,4)
    ey = np.rollaxis(np.asarray(ey),0,4)
    ez = np.rollaxis(np.asarray(ez),0,4)

    llentr = (llentr == 'y')
    if llentr:
        filename = f_path + 'Entropy.bin'
        time = []
        entr = []
        dentr = []
        if outformat == 0:
            with open(filename, 'rb') as file:
                file.read(4)
                for i in range(ns):
                    time.append(np.fromfile(file,dtype = "f8",count = 1,offset  = 0))

            #getting size of a0 to set an offset
                    nx,ny,nz = np.fromfile(file,dtype = "i4",count = 3,offset = 0)

                    TOTAL_SIZE = nx*ny*nz
                    SHAPE = (nx,ny,nz)
                    file.read(8)
                    entr.append(np.fromfile(file,dtype = "f8",count = TOTAL_SIZE,offset = 0).reshape(SHAPE,order = "F"))
                    dentr.append(np.fromfile(file,dtype = "f8",count = TOTAL_SIZE,offset = 0).reshape(SHAPE,order = "F"))
                    file.read(8)

    entr = np.asarray(entr)
    dentr = np.asarray(dentr)

    entr = np.rollaxis(entr,0,4)
    dentr = np.rollaxis(dentr,0,4)


    llpress = (llpress == 'y')
    if llpress:
        filename = f_path + 'Press.bin'
        time = []
        pxx = []
        pyy = []
        pzz = []
        pxy = []
        pxz = []
        pyz = []
        if outformat == 0:
            with open(filename, 'rb') as file:
                file.read(4)
                for i in range(ns):
                    time.append(np.fromfile(file,dtype = "f8",count = 1,offset  = 0))
            #getting size of a0 to set an offset
                    nx,ny,nz = np.fromfile(file,dtype = "i4",count = 3, offset = 0)

                    TOTAL_SIZE = nx*ny*nz
                    SHAPE = (nx,ny,nz)
                    file.read(8)
                    pxx.append(np.fromfile(file,dtype = "f8", count = TOTAL_SIZE, offset = 0).reshape(SHAPE,order = "F") )
                    pyy.append(np.fromfile(file,dtype = "f8", count = TOTAL_SIZE, offset = 0).reshape(SHAPE,order = "F") )
                    pzz.append(np.fromfile(file,dtype = "f8", count = TOTAL_SIZE, offset = 0).reshape(SHAPE,order = "F") )
                    pxy.append(np.fromfile(file,dtype = "f8", count = TOTAL_SIZE, offset = 0).reshape(SHAPE,order = "F") )
                    pxz.append(np.fromfile(file,dtype = "f8", count = TOTAL_SIZE, offset = 0).reshape(SHAPE,order = "F") )
                    pyz.append(np.fromfile(file,dtype = "f8", count = TOTAL_SIZE, offset = 0).reshape(SHAPE,order = "F") )
                    file.read(8)
                
    pxx = np.asarray(pxx)
    pyy = np.asarray(pyy)
    pzz = np.asarray(pzz)
    pxy = np.asarray(pxy)
    pxz = np.asarray(pxz)
    pyz = np.asarray(pyz)

    pxx = np.rollaxis(pxx,0,4)
    pyy = np.rollaxis(pyy,0,4)
    pzz = np.rollaxis(pzz,0,4)
    pxy = np.rollaxis(pxy,0,4)
    pxz = np.rollaxis(pxz,0,4)
    pyz = np.rollaxis(pyz,0,4)

    ns = dimt
    t = time

    return(ns,t,den,uxp,uyp,uzp,pxx,pyy,pzz,pxy,pxz,pyz,entr,dentr,bx,by,bz,ex,ey,ez)
