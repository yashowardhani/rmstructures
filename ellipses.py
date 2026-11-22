#!/usr/bin/env python
# coding: utf-8

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import h5py
import time
import argparse
import sys
from mpi4py import MPI
import glob

# get MPI ranks
comm = MPI.COMM_WORLD
nPE = comm.Get_size()
myPE = comm.Get_rank()
if nPE > 1:
    if (myPE == 0):
        print("ellipses: redirecting stdout and stderr to shell_???.out...", flush=True)
    myPE_str = f"{myPE:03d}"
    sys.stdout = open("shell_"+myPE_str+".out", 'a', buffering=1)
    sys.stderr = sys.stdout # stderr goes to same file

# ========================================================================
def print_mpi(string):
    string = str(string)
    comm = MPI.COMM_WORLD
    print("["+str(comm.Get_rank())+"] "+string)

# ========================================================================
def print_master(string):
    string = str(string)
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        print("["+str(comm.Get_rank())+"] "+string)

# ========================================================================
def get_3d_coords(n):
    vdel = 1/n
    vmin = vdel/2
    vmax = 1-vdel/2

    x = np.arange(vmin, vmax+vdel, vdel)
    y = np.arange(vmin, vmax+vdel, vdel)
    z = np.arange(vmin, vmax+vdel, vdel)
    yc, xc, zc = np.meshgrid(x, y, z)

    return xc, yc, zc

# ========================================================================
def rotation_vector(theta_x, theta_y, theta_z):
    R_x = np.array([[1, 0, 0],
                    [0, +np.cos(theta_x), -np.sin(theta_x)],
                    [0, +np.sin(theta_x), +np.cos(theta_x)]])
    R_y = np.array([[+np.cos(theta_y), 0, +np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, +np.cos(theta_y)]])
    R_z = np.array([[+np.cos(theta_z), -np.sin(theta_z), 0],
                    [+np.sin(theta_z), +np.cos(theta_z), 0],
                    [0, 0, 1]])
    return np.matmul(R_x, np.matmul(R_y, R_z))

# ========================================================================
def distance_array():
    # distance of points from ellipsoid centres
    dist = np.array([[x-x0, x-x0+1, x-x0-1],
                    [y-y0, y-y0+1, y-y0-1],
                    [z-z0, z-z0+1, z-z0-1]])

    # arg for the minimum distance
    argmin = np.array([np.argmin(np.abs(dist[0]), axis=0),
                    np.argmin(np.abs(dist[1]), axis=0),
                    np.argmin(np.abs(dist[2]), axis=0)])

    # minimum distance
    min_dist = np.array([dist[0][argmin[0], j_arr],
                        dist[1][argmin[1], j_arr],
                        dist[2][argmin[2], j_arr]])
    return min_dist

# ========================================================================
#to check if a function gives error
def iserror(func, *args, **kw):
    try:
        func(*args, **kw)
        return False
    except BaseException as error:
        return True


#=========================================================================
def save_stuff(now):
    if os.path.exists(npz_filename):
        os.system('cp %s %s/'%(npz_filename, backup_dir))
    time = now - start_time_model
    np.savez(npz_filename, mag=mag, ne=ne, mag_ind=mag_ind, ne_ind=ne_ind, vol_filled=vol_filled, ellipsoid_count=ellipsoid_count, time=time)


#=========================================================================
# MAIN
if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Create cubes of ellipsoids for magnetic field (mag) and electron density (ne).')
    parser.add_argument("-n", "--n", type=int, help="Number of cells (n^3) for generating the cubes (default: %(default)s)", default=200)
    parser.add_argument("-p", "--factor", type=float, help="Average aspect ratio of the ellipsoids", default=3)
    parser.add_argument("-m", "--m", type=float, help="Exponent on ne to set the B magnitude (B = ne^m) (default: %(default)s)", default=0.5)
    parser.add_argument("-M", "--M", type=float, help="Mach number (default: %(default)s)", default=2)
    parser.add_argument("-beta", "--beta", type=float, help="The beta in the sigma-mach number relationship (default: %(default)s)", default=0.4)
    parser.add_argument("-ff", "--fill_fractions", nargs="*", type=float, help="Volume filling factor (default: %(default)s)", default=[0.1, 0.2, 0.4, 0.6, 0.8])
    parser.add_argument("-seed", "--random_seed", type=int, help="Random seed for randomising ellipsoids (default: %(default)s)", default=300)
    parser.add_argument("-suf", "--suf", type=str, help="String appended at the end of model name (default: %(default)s)", default='')

    #checkpointing necessary for N=400 or higher on gadi
    parser.add_argument("-restart", "--restart", action='store_true', help="Whether to start from scratch instead of from saved chackpoints (default: %(default)s)", default=False)
    parser.add_argument("-cp", "--cp", action='store_true', help="To save checkpoints, or continue from checkpoints (default: %(default)s)", default=True)
    
    #supply disp and theta using a range or list
    parser.add_argument("-dtlist", "--dtlist", nargs="*", type=float, help="Disp-theta pairs to generate", default=[])

    # optional outname
    parser.add_argument("--outname", default=None)
    args = parser.parse_args()

    # start the timer
    start_time = round(time.time())

    # print MPI info
    print_master("Total number of MPI ranks = "+str(nPE))
    comm.Barrier()

    n = args.n
    factor = args.factor

    # an adjustment factor to ensure same average volume for ellipsoids of different aspect ratios
    adjust_size = (3/factor)**(1/3)
    ll, ul = 0.02*adjust_size, 0.04*adjust_size
    
    ln_sigma = (np.log(1 + args.M**2*args.beta**2))**0.5
    ln_mean = -ln_sigma**2/2
    # print info about parameter settings
    args.fill_fractions = sorted(args.fill_fractions)
    print_master("Cube size to be computed is "+str(args.n)+" (cubed).")
    print_master("Target filling fractions are "+str(args.fill_fractions)+".")

    if not args.restart:    print_master("Continuing from saved checkpoints (if available).")
    else:                   print_master("Ignoring saved checkpoints and starting from scratch!")

    # set up coordinates
    x, y, z = get_3d_coords(args.n)
    x = np.ravel(x)
    y = np.ravel(y)
    z = np.ravel(z)

    # assign disps and thetas
    if len(args.dtlist):
        disps = args.dtlist[::2]
        thetas_deg = np.array(args.dtlist[1::2], dtype=int)
        thetas = thetas_deg*np.pi/180
        n_total = len(disps)
    else:
        disps = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.5]
        thetas_deg = [0, 30, 60, 90]
        thetas = np.array(thetas_deg)*np.pi/180
    
        # number of displacements, angles, and total number of models
        n_disps = len(disps)
        n_thetas = len(thetas)
        n_total = n_disps * n_thetas

    # list of displacements to process
    if np.max(disps) > 0.5:
        print_master("Warning: maximum displacement set to a value > 0.5 (half the box size); not useful for periodic cubes...")
        exit()


    # printing disp and theta info
    print_master(' ')
    print_master("computing models for")
    print_master("thetas:")
    print_master(thetas_deg)    # print in degrees, not radian
    print_master("displacements:")
    print_master(disps)
    print_master("thus, we are computing a total number of "+str(n_total)+" models.")

    # domain decomposition
    # first check for reasonable number of cores
    if n_total % nPE != 0:
        print_master('ERROR: number of compute cores should be integer multiple of total number of models. Exiting.')
        exit()
    my_start = myPE * (n_total // nPE)
    my_end = (myPE+1) * (n_total // nPE) - 1
    # last PE gets the rest
    if (myPE == nPE-1):
        my_end = n_total - 1
    print_mpi("my_start = "+str(my_start)+", my_end = "+str(my_end))

    # list of datasets saved at even checkpoints as npy files
    chk_datasets = ['mag', 'ne', 'mag_ind', 'ne_ind', 'vol_filled', 'ellipsoid_count', 'time']

    # main loop over all models (MPI parallelised)
    for it in range(my_start, my_end+1):
        start_time_model = round(time.time())

        if len(args.dtlist):
            disp = disps[it]
            theta = thetas[it]
            theta_deg = thetas_deg[it]
        else:
            # the total index would be constructed like this: it = j*n_disps + i, and thus:
            i = it % n_disps # index into disps
            j = it // n_disps # index into thetas

            disp = disps[i] # this displacement for this proc
            theta = thetas[j] # this angle for this proc
            theta_deg = thetas_deg[j]

        print_master(' ')
        print_mpi('working on displacement '+str(disp)+" and angle "+str(theta_deg)+" deg...")

        # global ellipsoid indices and arrays
        mag_ind = np.array([])
        ne_ind = np.array([])
        mag = np.zeros((args.n, args.n, args.n))
        ne = np.zeros((args.n, args.n, args.n))

        # and more things back to 0
        vol_filled = [0, 0]
        ellipsoid_count = 0

        # check how many fill fractions already present
        if args.restart:
            ff_missing = args.fill_fractions
            filenames = ['ellipsoids_f%gd%gt%g%s.h5' % (ffn, disp, theta_deg, args.suf) for ffn in args.fill_fractions]
            filenames.extend('backups_%g/d%g_t%d.npz'%(ffn, disp, theta) for ffn in args.fill_fractions)
            filenames.append('d%g_t%d.npz'%(disp, theta))
            for filename in filenames:
                try:    os.remove(filename)
                except: continue
        else:
            ff_missing = []
            for ffn in args.fill_fractions:
                outname = 'ellipsoids_f%gd%gt%g%s.h5' % (ffn, disp, theta_deg, args.suf)
                if not os.path.exists(outname):
                    ff_missing.append(ffn)
                else:
                    print_mpi('%s already present. skipping'%outname)
            if not len(ff_missing): continue

        # loading a saved checkpoint
        npz_filename = 'd%g_t%g.npz'%(disp, theta_deg)
        if not args.restart:
            data = None
            # loading backup file if present
            # check for backup in current directory
            try:
                data = np.load(npz_filename)
            except:
                # load backup from a backup directory with largest but smaller ff
                backup_dirs = sorted(glob.glob('backups_*'), reverse=True)
                if len(backup_dirs):
                    backed_ff_dirs = np.array([float(x.split('_')[1]) for x in backup_dirs])
                    avail_backups = backed_ff_dirs[backed_ff_dirs<ffn]
                    for backed_ff in avail_backups:
                        backup = 'backups_%g/%s'%(backed_ff, npz_filename)
                        if os.path.exists(backup):
                            try:
                                os.system('cp %s .'%backup)
                                data = np.load(npz_filename)
                                break
                            except:
                                continue

            if data is not None:
                start_time_model = start_time_model - float(data['time'])
                mag = data['mag']
                ne = data['ne']
                mag_ind = list(data['mag_ind'])
                ne_ind = list(data['ne_ind'])
                vol_filled = data['vol_filled']
                ellipsoid_count = int(data['ellipsoid_count'])

        # picking random seed for each ellipsoiod from an array of seeds that is itself generated using the supplied random seed
        seed_array_file = 'seeds_s%s.npy'%args.random_seed
        if not os.path.exists(seed_array_file):
            np.random.seed(args.random_seed)
            single_ellipsoid_seeds = np.unique(np.random.choice(10000000, 100000))
            np.save(seed_array_file, single_ellipsoid_seeds)
        else:
            single_ellipsoid_seeds = np.load(seed_array_file, allow_pickle=True)

        # iterating through the list of missing ffs
        ff_iter = iter(ff_missing)
        ffn = next(ff_iter)
        stop_adding_ellipsoids = False
        n_grid = n**3
        j_arr = np.arange(n_grid)
        stone = start_time_model

        # backup dir for checkpoint npz files
        backup_dir = 'backups_%g'%ffn
        os.makedirs(backup_dir, exist_ok=True)

        while True:
            # SAVE H5 FILE WHEN THE REQUISITE FF IS REACHED
            if np.min(vol_filled) > ffn:
                end_time_model = round(time.time())
                save_stuff(end_time_model)
                # save 3D cubes
                if args.outname is not None:
                    outname = args.outname
                else:
                    outname = 'ellipsoids_f%gd%gt%g%s.h5' % (ffn, disp, theta_deg, args.suf)
                print_mpi("saving file '" + outname + "'...")
                with h5py.File(outname, 'w') as hf:
                    hf.create_dataset('mag',    data=np.swapaxes(mag/np.std(mag[mag!=0]), 0, 2))
                    hf.create_dataset('ne',     data=np.swapaxes(ne/np.mean(ne[ne!=0]), 0, 2))
                    hf.create_dataset('time',   data=end_time_model - start_time_model)
                    hf.create_dataset('ellipsoid_count', data=ellipsoid_count)
                    hf.create_dataset('mach',   data=args.M)
                    hf.create_dataset('beta',   data=args.beta)
                # stop iterating once all missing ffs are found
                try:    ffn = next(ff_iter)
                except: break
            
            # CREATE A SINGLE ELLIPSOID
            np.random.seed(single_ellipsoid_seeds[ellipsoid_count])

            # ellipsoid location and size
            x0, y0, z0 = np.random.uniform(0, 1, 3) # random centre for ellipsoid 
            a, b = np.random.uniform(ll, ul, 2) # ellipsoid parameters
            c = np.random.uniform(factor*ll, factor*ul)
            # ellipse matrix
            M = np.diag([a**-2, b**-2, c**-2])

            # draw random x,y,z rotation angles (0, 360 deg)
            randangle_x = np.random.uniform(0, np.pi)
            randangle_y = np.random.uniform(0, np.pi)
            randangle_z = np.random.uniform(0, np.pi)
            R = rotation_vector(randangle_x, randangle_y, randangle_z)
            M_ne = np.matmul(R.T, np.matmul(M, R))  # combined array 

            # fetching indices of points within a major axis cube only
            distance = distance_array()
            ind = np.where(np.all(np.abs(distance) < c, axis=0))[0]
            dist_reduced = distance.T[ind].T

            # getting indices of points within ellipsoid
            ind_ellipse = np.where(np.einsum('ij,ji->i', np.matmul(dist_reduced.T, M_ne), dist_reduced) <= 1)
            new_index = ind[ind_ellipse]
            ne_ind = np.union1d(ne_ind, new_index)

            # filling points with value picked from lognormal distribution
            # mean and sigma of underlying normal dist calculated using mach number
            new_ne_value = np.random.lognormal(ln_mean, ln_sigma)
            ne[np.unravel_index(new_index, (n, n, n))] += new_ne_value

            # MAG ELLIPSOID
            # adjusting rotation vector
            match np.argmin([a, b]):
                case 0: randangle_x += theta
                case 1: randangle_y += theta
            R_mag = rotation_vector(randangle_x, randangle_y, randangle_z)
            M_mag = np.matmul(R_mag.T, np.matmul(M, R_mag))

            # adjusting ellipsoid centre position
            randangle_theta = np.arccos(np.random.uniform(-1, 1))
            randangle_phi = np.random.uniform(0, 2*np.pi)
            x0 += disp*np.sin(randangle_theta)*np.cos(randangle_phi)
            y0 += disp*np.sin(randangle_theta)*np.sin(randangle_phi)
            z0 += disp*np.cos(randangle_theta)

            # fetching indices of points within a major axis cube only
            distance = distance_array()
            ind = np.where(np.all(np.abs(distance) < c, axis=0))[0]
            dist_reduced = distance.T[ind].T

            # getting indices of points within ellipsoid
            ind_ellipse = np.where(np.einsum('ij,ji->i', np.matmul(dist_reduced.T, M_mag), dist_reduced) <= 1)
            new_index = ind[ind_ellipse]
            mag_ind = np.union1d(mag_ind, new_index)

            # filling points with the ne_ellipsoid value ^ m
            # fiving a random sign
            new_mag_value = new_ne_value**args.m*np.random.choice([-1,1])
            mag[np.unravel_index(new_index, (n, n, n))] += new_mag_value

            # calculating volume filling fraction
            vol_filled = [len(mag_ind) / n_grid, len(ne_ind) / n_grid] 
            ellipsoid_count += 1 # increment count

            print_mpi("%g, %d; vol_fill: %g, %g, ellipsoids: %g"%(disp, theta_deg, vol_filled[0], vol_filled[1], ellipsoid_count))

            # SAVE CHECKPOINT FILES
            if args.cp:
                now = round(time.time())
                # save every x seconds
                if (now - stone) >= 300:
                    save_stuff(now)
                    stone = now # increment milestone for the next round of saving

    # print total script runtime
    end_time = round(time.time())
    print_mpi("Runtime: "+str(end_time-start_time)+"s")