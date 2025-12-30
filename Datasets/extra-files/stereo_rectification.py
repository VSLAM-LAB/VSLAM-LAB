
import numpy as np
import cv2

fx0, fy0, cx0, cy0 = 728.7329, 729.0125, 626.0223, 531.8843
fx1, fy1, cx1, cy1 = 731.4918, 731.7279, 614.8181, 514.6389
D0= np.array([-0.0463, 0.1427, 0.00067775, -0.00082188, -0.089])
D1 = np.array([-0.0491, 0.1516, 0.00027016, -0.00015604, -0.0955])
size = (1224, 1024)
   
K0 = np.array([[fx0, 0.,     cx0],
                [0.,     fy0, cy0],
                 [0.,     0.,     1. ]], dtype=np.float64)
K1 = np.array([[fx1, 0.,     cx1],
                [0.,     fy1, cy1],
                [0.,     0.,     1. ]], dtype=np.float64)

# T_BS0 = np.array(cam0["T_BS"]["data"], dtype=float).reshape(cam0["T_BS"]["rows"], cam0["T_BS"]["cols"])
# T_BS1 = np.array(cam1["T_BS"]["data"], dtype=float).reshape(cam1["T_BS"]["rows"], cam1["T_BS"]["cols"])
# R_SB0, t_SB0 = T_BS0[:3, :3], T_BS0[:3, 3]
# R_SB1, t_SB1 = T_BS1[:3, :3], T_BS1[:3, 3]
# R_01_B = R_SB1.T @ R_SB0
# t_01_B = R_SB1.T @ (t_SB0 - t_SB1)

#R_01_B = np.array([0.9992, -0.0071, 0.0397, 0.0061,    0.9997,    0.0242, -0.0399,   -0.0239,    0.9989 ]).reshape(3,3)
#t_01_B = np.array([ -0.1969122, -0.00005657, 0.0057118 ])
R_01_B = np.array([0.9992,   -0.0070,    0.0398, 0.0060,    0.9996,    0.0260, -0.0400,   -0.0257,    0.9989]).reshape(3,3)
t_01_B = np.array([ -0.1962696, -0.0042513, 0.0047504 ])

R_l, R_r, P_l, P_r, Q, _, _ = cv2.stereoRectify(
    K0, D0, K1, D1, size, R_01_B.astype(np.float64), t_01_B.astype(np.float64),
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0, newImageSize=size
)

print(P_l)
print(P_r)
print(P_r[0,3] / P_r[0,0])  # baseline in meters