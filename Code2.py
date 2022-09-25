from robodk import *
import robolink as rl    # RoboDK API
import robodk as rdk     # Robot toolbox
import numpy as np
import math


"""FIND: Function to search through all the possible joint angles that return a certian pose
Use it to ensure right-handedness to set all intermediate joint points"""


RDK = rl.Robolink()

robot = RDK.Item('UR5')
world_frame = RDK.Item('UR5 Base')
target = RDK.Item('Home')   # existing target in station
robot.setPoseFrame(world_frame)
robot.setPoseTool(robot.PoseTool())

def getRotation(oPa, oPb, aPb):
    """Returns the rotation matrix of coordinate frame A relative to
    the origin O, using point B"""
    # oPa means position of a w.r.t o
    # aPbINV = np.transpose(1/(np.matmul(np.transpose(aPb),aPb))*aPb)
    # oRa = np.matmul(aPbINV, (oPb-oPa))
    aPbNorm = aPb/np.linalg.norm(aPb)
    alpha = math.acos(aPbNorm[1])
    if aPbNorm[0] < 0:
        alpha = -alpha
    v = oPb - oPa
    vNorm = v/np.linalg.norm(v)
    beta = math.acos(vNorm[1])
    if vNorm[0] < 0:
        beta = -beta
    # print(aPbNorm,vNorm)
    theta = alpha - beta
    sinTheta = math.sin(theta)
    cosTheta = math.cos(theta)
    oRa = np.array([[cosTheta, -sinTheta, 0],
                    [sinTheta, cosTheta, 0],
                    [0, 0, 1]])
    theta = math.atan2(sinTheta, cosTheta)*180/math.pi
    # theta = theta*180/math.pi
    return oRa, theta

def getTransform(aRb, aPb):
    """Transform to convert a vector in terms of A to terms of O"""
    aTb = [[aRb[0][0], aRb[0][1], aRb[0][2], aPb[0][0]], 
        [aRb[1][0], aRb[1][1], aRb[1][2], aPb[1][0]],
        [aRb[2][0], aRb[2][1], aRb[2][2], aPb[2][0]],
        [0, 0, 0, 1]]
    return aTb

# TOOLS
Identity = np.array([[1, 0, 0], [0, 1, 0],[0, 0, 1]])
gToolTheta = -50/180*np.pi
R_grinder = np.array([[math.cos(gToolTheta), -math.sin(gToolTheta), 0.0], [math.sin(gToolTheta), math.cos(gToolTheta), 0.0], [0.0, 0.0, 1.0]])
P_grinderPush = np.transpose(np.array([[0.0, 0.0, 102.82]]))
T_grinderPush = rdk.Mat(np.linalg.inv(getTransform(R_grinder, P_grinderPush)).tolist())
# tcpTgt = np.linalg.inv(getTransform(R_grinder, np.transpose(np.array([[0, 0, 0, 0]]))))
# gtTpb = np.linalg.inv(getTransform(Identity, P_grinderPush))
# tcpTpb = np.matmul(gtTpb, tcpTgt)
# print(T_grinderPush)
# print(tcpTpb)
# print(T_grinderPush)
# print("T_grinderPush = ", T_grinderPush)

### TOOL FRAME ###
# Vectors
oPt = np.transpose(np.array([[-555.6, -78.5, 19.05]]))
oPq = np.transpose(np.array([[-645.0, 77.2, 19.05]]))
tPq = np.transpose(np.array([[-127.0, 127.0, 0]]))
gtaP = np.transpose(np.array([[144.0, -98.6, 515.0]]))
ptaP = np.transpose(np.array([[144.0, 67.0, 515.0]]))
ctaP = np.transpose(np.array([[144.0, 231.5, 515.0]]))
portafilterEntryP = np.transpose(np.array([[9.5,67.3,214]]))
# print(getRotation(oPt, oPq, tPq))
# Matrices
oRt, thetaT = getRotation(oPt, oPq, tPq)
T_TF = rdk.Mat(getTransform(oRt, gtaP))
# print(T_grinderApproach)
# T_portafilterApproach = rdk.Mat(getTransform(oRt, ptaP))
# T_cupApproach = rdk.Mat(getTransform(oRt, ctaP))
# T_portafilterEntryApproach = rdk.Mat(getTransform(oRt, portafilterEntryP))
# print(oPt + np.matmul(oRt, tPq), "==\n", oPq, '\n') #GOOD

### COFFEE MACHINE FRAME
oPc = np.transpose(np.array([[-365.5, -386.7, 349.8]]))
oPq = np.transpose(np.array([[-577.8, -441.6, 349.8]]))
cPq = np.transpose(np.array([[0.0, 218.0, 0.0]]))
oRc, thetaC = getRotation(oPc, oPq, cPq)
T_CMF = rdk.Mat(getTransform(oRc, oPc))
T_CMF2 = rdk.Mat(np.matmul(T_CMF, T_grinderPush).tolist())
# print(T_CMF)
# Button 1: urTtcp = urTcm cmTbut1 inv(gtTpb) inv(tcpTgt)
P_cBut1 = np.transpose(np.array([[50.67,35.25,-27.89]]))
R_cBut1 = np.array([[0, 0, -1],[0, 1, 0],[1, 0, 0]])
# R_cBut1 = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
T_cBut1 = rdk.Mat(getTransform(R_cBut1, P_cBut1))
T_But1 = np.matmul(T_CMF, T_cBut1)
T_But1Target = T_CMF

# T_But1Target = rdk.Mat(np.matmul(T_CMF, tcpTgt).tolist())
# urTtcp = np.matmul(np.matmul(T_But1, gtTpb), tcpTgt)
# print(T_But1Target)
# print(urTtcp)
P_cBut2 = np.transpose(np.array([[50.67,35.25,-61.39]]))
P_cBut3 = np.transpose(np.array([[50.67,35.25,-94.89]]))
P_cSwtch1 = np.transpose(np.array([[50.67,98.75, -27.89]]))
P_cCupSpot= np.transpose(np.array([[-12.68,72.0,-290]]))

# print(target)
# T_TFR = rdk.Mat(tcpTgt.tolist())
print(T_But1Target)
# MOVEMENTS
RDK.setSimulationSpeed(2)
robot.MoveJ(target, blocking=True)
robot.MoveJ(T_But1Target, blocking=True)
# robot.MoveJ(T_TF, blocking=True)
# RDK.RunProgram("Grinder Tool Attach (Tool Stand)", True)
# RDK.setSimulationSpeed(0.3)
# robot.MoveJ(target, blocking=True)
