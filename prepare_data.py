import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Diffusion Model at 32x32')
parser.add_argument('-type', '--type', type=str, metavar='', help='type of example')
args = parser.parse_args()

if __name__ == "__main__":
    if args.type:
        print('User defined problem')
        type = args.type
    else:
        print('Not define problem type, use default poentential vorticity')
        type = 'PV'
        # type = 'WD'

    cwd = os.getcwd()
    ### define problem type ##########
    if type == 'PV':
        from config.config_pv import *
    if type == 'WD':
        from config.config_wd import *
    elif type == 'NS':
        from config.config_ns import *
    elif type == 'ELAS0':
        from config.config_elas0 import *
    elif type == 'ELAS1':
        from config.config_elas1 import *
    elif type == 'ELAS2':
        from config.config_elas2 import *

    ########################## gen data ##########################
    with open(Ori_data_name_high, 'rb') as ss:
        mat_f = np.load(ss)

    if mat_f.ndim == 4:
        mat_f = mat_f[:, :, :, 0]


    mat_f = make_image(mat_f)
    mat_f= mat_f[:N_gen, ..., None]

    sp = mat_f[:, ::m, ::m, :]
    sp = make_image(sp)

    cv1_0, cv1_1, cv1_2 = prepare_cv_data(kernel=1, blur=1, mat=mat_f)
    cv1_0 = make_image(cv1_0)

    cv2_0, cv2_1, cv2_2 = prepare_cv_data(kernel=2, blur=1, mat=mat_f)
    cv2_0 = make_image(cv2_0)

    with open(Gen_data, 'wb') as ss:
        np.save(ss, sp)
        np.save(ss, cv1_0)
        np.save(ss, cv2_0)

    ################################ Sup data ################################
    mat_sp_u_0, mat_sp_u_1, mat_sp_u_2, mat_sp_u_d = prepare_up_skip(L, points_x_0, points_x_1, points_x_2,
                                                                     points_x, mat_f)

    mat1_u_0, mat1_u_1, mat1_u_2, mat1_u_d = prepare_up_cv(L, points_x_0, points_x_1, points_x_2, points_x, mat_f,
                                                           kernel=1, blur=1)

    mat2_u_0, mat2_u_1, mat2_u_2, mat2_u_d = prepare_up_cv(L, points_x_0, points_x_1, points_x_2, points_x, mat_f,
                                                           kernel=2, blur=1)

    with open(Super_data, 'wb') as ss:
        np.save(ss, mat_sp_u_0)
        np.save(ss, mat_sp_u_1)
        np.save(ss, mat_sp_u_2)

        np.save(ss, mat1_u_0)
        np.save(ss, mat1_u_1)
        np.save(ss, mat1_u_2)

        np.save(ss, mat2_u_0)
        np.save(ss, mat2_u_1)
        np.save(ss, mat2_u_2)

        np.save(ss, mat_f)

    ##### prepare for testing
    test_ls = []
    with open(Ori_data_name_test, 'rb') as ss:
        while True:
            try:
                test_ls.append(np.load(ss))  # Load each array
            except ValueError:
                break

    tmat_f = test_ls.pop()

    if tmat_f.ndim == 4:
        tmat_f = tmat_f[:, :, :, 0]

    tmat_f = make_image(tmat_f)
    tmat_f = tmat_f[..., None]

    sp = tmat_f[:, ::m, ::m, :]
    sp = make_image(sp)

    cv1_0, cv1_1, cv1_2 = prepare_cv_data(kernel=1, blur=1, mat=tmat_f)
    cv1_0 = make_image(cv1_0)

    cv2_0, cv2_1, cv2_2 = prepare_cv_data(kernel=2, blur=1, mat=tmat_f)
    cv2_0 = make_image(cv2_0)

    mat_sp_u_0, mat_sp_u_1, mat_sp_u_2, mat_sp_u_d = prepare_up_skip(L, points_x_0, points_x_1, points_x_2,
                                                                     points_x, tmat_f)

    mat1_u_0, mat1_u_1, mat1_u_2, mat1_u_d = prepare_up_cv(L, points_x_0, points_x_1, points_x_2, points_x, tmat_f,
                                                           kernel=1, blur=1)

    mat2_u_0, mat2_u_1, mat2_u_2, mat2_u_d = prepare_up_cv(L, points_x_0, points_x_1, points_x_2, points_x, tmat_f,
                                                           kernel=2, blur=1)


    with open(Test_data, 'wb') as ss:
        ### save the low
        for array in test_ls:
            np.save(ss, array)

        np.save(ss, tmat_f)

        ### save at 32x32
        np.save(ss, sp)
        np.save(ss, cv1_0)
        np.save(ss, cv2_0)
        ### save the high and it's relevant
        np.save(ss, mat_sp_u_0)
        np.save(ss, mat_sp_u_1)
        np.save(ss, mat_sp_u_2)

        np.save(ss, mat1_u_0)
        np.save(ss, mat1_u_1)
        np.save(ss, mat1_u_2)

        np.save(ss, mat2_u_0)
        np.save(ss, mat2_u_1)
        np.save(ss, mat2_u_2)

