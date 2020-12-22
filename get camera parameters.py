
import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2


def calibrate_single(imgNums, CheckerboardSize, Nx_cor, Ny_cor, saveFile=False, saveImages=False):

    '''单目(普通+广角/鱼眼)摄像头标定
mtx,dist,K,D = calibrate_single(30,27,8,6,True,True)
    :param imgNums: 标定所需样本数,一般在20~40之间.按键盘空格键实时拍摄

    :param CheckerboardSize: 标定的棋盘格尺寸,必须为整数.(单位:mm或0.1mm)

    :param Nx_cor: 棋盘格横向内角数

    :param Ny_cor: 棋盘格纵向内角数

    :param saveFile: 是否保存标定结果,默认不保存.

    :param saveImages: 是否保存图片,默认不保存.

    :return mtx: 内参数矩阵.{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}

    :return dist: 畸变系数.(k_1,k_2,p_1,p_2,k_3)'''

    # 找棋盘格角点(角点精准化迭代过程的终止条件)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, CheckerboardSize, 1e-6)  # (3,27,1e-6)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE  # 11

    flags_fisheye = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW  # 14

    # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((1, Nx_cor * Ny_cor, 3), np.float32)

    objp[0, :, :2] = np.mgrid[0:Nx_cor, 0:Ny_cor].T.reshape(-1, 2)

    # 储存棋盘格角点的世界坐标和图像坐标对

    objpoints = []  # 在世界坐标系中的三维点

    imgpoints = []  # 在图像平面的二维点

    count = 0  # 用来标志成功检测到的棋盘格画面数量

    
    imageSize=(640,480)
    while (True):
        _, frame = cap.read()
        frame = cv2.resize(frame, imageSize)
        
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):#在等待的1s之内按下了空格键则执行
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('gray',gray)      

            # print('寻找棋盘格模板的角点',Nx_cor,Ny_cor)
            ok, corners = cv2.findChessboardCorners(gray, (Nx_cor, Ny_cor), flags)

            if count >= imgNums:
                break
            print(ok,' sample')
            if ok:  # 如果找到，添加目标点，图像点

                objpoints.append(objp)

                cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 获取更精确的角点位置

                imgpoints.append(corners)

                # 将角点在图像上显示
                cv2.drawChessboardCorners(frame, (Nx_cor, Ny_cor), corners, ok)
                count += 1
                if saveImages:
                    cv2.imwrite('/home/dzf/FishCamera/test_imgs/' + str(count) + '.jpg', frame)

                print('NO.' + str(count))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    global mtx, dist

    # 标定. rvec和tvec是在获取了相机内参mtx,dist之后通过内部调用solvePnPRansac()函数获得的

    # ret为标定结果，mtx为内参数矩阵，dist为畸变系数，rvecs为旋转矩阵，tvecs为平移向量

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[:2][::-1], None, criteria)

    # 摄像头内参mtx = [[f_x,0,c_x][0,f_y,c_y][0,0,1]]
    # print('mtx=np.array( ' + str(mtx.tolist()) + " )")

    # 畸变系数dist = (k1,k2,p1,p2,k3)
    # print('dist=np.array( ' + str(dist.tolist()) + " )")

    # 鱼眼/大广角镜头的单目标定
    K = np.zeros((3, 3))#返回一个用0填充的3x3的数组，来初始化K
    D = np.zeros((4, 1))
    RR = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
    TT = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
    rms, _, _, _, _ = cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[:2][::-1], K, D, RR, TT, flags_fisheye, criteria)

    # 摄像头内参,此结果与mtx相比更为稳定和精确
    # print("K=np.array( " + str(K.tolist()) + " )")
    # 畸变系数D = (k1,k2,k3,k4)
    # print("D=np.array( " + str(D.tolist()) + " )")

    # 计算反投影误差
    mean_error = 0

    for i in range(len(objpoints)):

        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)

        mean_error += error

    print("total error: ", mean_error / len(objpoints))

    if saveFile:

        # np.savez("/home/dzf/FishCamera/imgs/calibrate.npz", mtx=mtx, dist=dist, K=K, D=D)
        np.savetxt("/home/dzf/FishCamera/test_parameters_k_d/K.txt", K,fmt='%f',delimiter=' ')
        np.savetxt("/home/dzf/FishCamera/test_parameters_k_d/D.txt", D,fmt='%f',delimiter=' ')

    cv2.destroyAllWindows()

    return mtx, dist, K, D

if __name__ == '__main__':

    cam = 0
    cap=cv2.VideoCapture(cam)
    # cap=cv2.VideoCapture(cam,cv2.CAP_DSHOW)#电脑自带摄像头为0,usb外接摄像头为1 最好ls /dev查看
    ##cv2.CAP_DSHOW 参数初始化摄像头,否则无法使用更高分辨率
    mtx,dist,K,D = calibrate_single(30,27,8,6,True,True)
    print('mtx=',mtx)
    print('dist=',dist)
    print('K=',K)
    print(type(K))
    print('D=',D)
    width,height= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #优化内参教和畸变系数
    P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K,D,(width,height),None)

    #此处计算花费时间较大，需从循环中抽取出来
    mapx2,mapy2=cv2.fisheye.initUndistortRectifyMap(K,D,None,P,(width,height),cv2.CV_32F)
    print('mapx2=',mapx2)
    print('mapy2=',mapy2)

    while (True):
        ret,frame = cap.read()
        cv2.imshow('raw',frame)

        #畸变矫正
        frame_rectified = cv2.remap(frame,mapx2,mapy2,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

        cv2.namedWindow("RectifiedImage",cv2.WINDOW_FREERATIO)
        cv2.resizeWindow("RectifiedImage",1280,960)#调整显示窗口大小
        cv2.imshow('RectifiedImage',frame_rectified)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    cap.release()
    cv2.destroyAllWindows()