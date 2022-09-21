from robodk import *
import robolink as rl    # RoboDK API
import robodk as rdk     # Robot toolbox
import numpy as np
import math

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
    # theta = math.atan2(sinTheta, cosTheta)*180/math.pi
    theta = theta*180/math.pi
    return oRa

def getTransform(aRb, aPb):
    """Transform to convert a vector in terms of A to terms of O"""
    aTb = [[aRb[0][0], aRb[0][1], aRb[0][2], aPb[0][0]], 
        [aRb[1][0], aRb[1][1], aRb[1][2], aPb[1][0]],
        [aRb[2][0], aRb[2][1], aRb[2][2], aPb[2][0]],
        [0, 0, 0, 1]]
    return aTb


# Directly use the RDK Matrix object from to hold pose (its an HT)
T_home = rdk.Mat([[     0.000000,     0.000000,     1.000000,   523.370000 ],
     [-1.000000,     0.000000,     0.000000,  -109.000000 ],
     [-0.000000,    -1.000000,     0.000000,   607.850000 ],
      [0.000000,     0.000000,     0.000000,     1.000000 ]])

### GLobal Frame
P_PF2 = np.transpose(np.array([[370.5, -322.5, 65.9]]))
oRo = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

### TOOL Transforms ###
Identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
P_grinderPush = np.transpose(np.array([[0.0, 0.0, 102.82]]))
T_grinderPush = getTransform(Identity, P_grinderPush)

portafilterBearingP = np.transpose(np.array([[-32.0, 0.0, 27.56]]))
portafilterCenterP = np.transpose(np.array([[4.71, 0.0, 144.76]]))
R_portafilterToolToGrinder = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
portafilterToolBearingAddition = np.matmul(R_portafilterToolToGrinder,portafilterBearingP)
# print(portafilterToolBearingAddition)

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
oRt = getRotation(oPt, oPq, tPq)
T_grinderApproach = rdk.Mat(getTransform(oRt, gtaP))
# print(T_grinderApproach)
T_portafilterApproach = rdk.Mat(getTransform(oRt, ptaP))
T_cupApproach = rdk.Mat(getTransform(oRt, ctaP))
T_portafilterEntryApproach = rdk.Mat(getTransform(oRt, portafilterEntryP))
# print(oPt + np.matmul(oRt, tPq), "==\n", oPq, '\n') #GOOD

### GRINDER FRAME ###
# Vectors
oPg = np.transpose(np.array([[482.7, -432.1, 316.1]]))
oPqg = np.transpose(np.array([[370.5, -322.5, 65.9]]))
gPq = np.transpose(np.array([[157.61, 0.0, -250.45]]))
oRg = getRotation(oPg, oPqg, gPq)
# print(oRg)
# P_gPF2 = np.transpose(np.array([[157.61, 0.0, -250.45]]))
P_PF2target = gPq + portafilterToolBearingAddition
P_PF2approach = P_PF2target + np.transpose(np.array([[10.0, 0.0, 10.0]]))
P_gSlider = np.transpose(np.array([[-35.82, 83.8, -153]]))
P_gButOn = np.transpose(np.array([[-64.42, 89.82, -227.68]]))
P_gButOff = np.transpose(np.array([[-80.71, 94.26, -227.68]]))
# print(getRotation(oPg, oPq, gPq))
# Matrices
PF2T = rdk.Mat(getTransform(oRg, gPq))
T_gSlider = rdk.Mat(getTransform(oRg, P_gSlider))
T_gButOn = rdk.Mat(getTransform(oRg, P_gButOn))
T_gButOff = rdk.Mat(getTransform(oRg, P_gButOff))
# T_PF2target = rdk.Mat(getTransform(oRg, P_PF2target))
T_PF2approach = rdk.Mat(getTransform(oRg, P_PF2approach))
T_PF2target = rdk.Mat(getTransform(oRo, P_PF2))
# print(oPg + np.matmul(oRg, gPq), "==\n", oPqg, '\n') #BAD

# ### COFFEE MACHINE FRAME
# Vectors
oPc = np.transpose(np.array([[-365.5, -386.7, 349.8]]))
oPq = np.transpose(np.array([[-577.8, -441.6, 349.8]]))
cPq = np.transpose(np.array([[0.0, 218.0, 0.0]]))
oRc = getRotation(oPc, oPq, cPq)
T_CMF = getTransform(oRc, oPc)
T_CMF = rdk.Mat(np.matmul(T_CMF, np.linalg.inv(T_grinderPush)).tolist())
# T_CMF = rdk.Mat(T_CMF)
# print(T_CMF)
P_cBut1 = np.transpose(np.array([[50.67,35.25,-27.89]]))
P_cBut2 = np.transpose(np.array([[50.67,35.25,-61.39]]))
P_cBut3 = np.transpose(np.array([[50.67,35.25,-94.89]]))
P_cSwtch1 = np.transpose(np.array([[50.67,98.75, -27.89]]))
P_cCupSpot= np.transpose(np.array([[-12.68,72.0,-290]]))
# print(getRotation(oPc, oPq, cPq))
# Matrices
T_cBut1 = rdk.Mat(getTransform(oRc, P_cBut1))
T_cBut2 = rdk.Mat(getTransform(oRc, P_cBut2))
T_cBut3 = rdk.Mat(getTransform(oRc, P_cBut3))
T_cSwtch1 = rdk.Mat(getTransform(oRc, P_cSwtch1))
T_cCupSpot = rdk.Mat(getTransform(oRc, P_cCupSpot))
# print(oPc + np.matmul(oRc, cPq), "==\n", oPq, '\n')
grinderToolPush = np.array([[0, 0, 102.82]])
P_cBut1App = P_cBut1 + np.array([[112.82, 0, 0]])
T_cBut1App = rdk.Mat(getTransform(oRc, P_cBut1App))

# ### SCRAPER FRAME ###
# Vectors
oPs = np.transpose(np.array([[599.9, 53.0, 254.4]]))
oPq = np.transpose(np.array([[582.7, 128.6, 235.8]]))
sPq = np.transpose(np.array([[-80.0, 0.0, 0.0]]))
P_sTamper = np.transpose(np.array([[-80.0, 0.0, -55.0]]))
P_sScraper = np.transpose(np.array([[70.0,0.0,-32.0]]))
# print(getRotation(oPs, oPq, sPq))
# # Matices
# oRs = getRotation(oPs, oPq, sPq)
# T_sTamper = rdk.Mat(getTransform(oRs, P_sTamper))
# T_sScraper = rdk.Mat(getTransform(oRs, P_sScraper))


# ### COFFEE CUP STAND FRAME ###
# # Vectors
# P_csCuplip = np.transpose(np.array([[-40.0,0.0,180.0]]))
# P_csCntre = np.transpose(np.array([[0.0,0.0,217.0]]))
# # Matices
# oRcs = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
# T_csCuplip = rdk.Mat(getTransform(oRcs, P_csCuplip))
# T_csCntre = rdk.Mat(getTransform(oRcs, P_csCntre))

# # Joint angles
J_intermediatepoint = [-151.880896, -97.616411, -59.103383, -112.890980, 90.242082, -161.879346]
J_intermediateTF2G = [-57.200000, -90.840000, -70.650000, -113.440000, 89.700000, -177.280000]

####### MOTION ############
# print(type(T_CMF))
# print(type(T_grinderApproach))
# print(T_CMF)
# print(T_grinderApproach)
# Push coffee machine buttons
RDK.setSimulationSpeed(2)
robot.MoveJ(target, blocking=True)
robot.MoveJ(T_grinderApproach, blocking=True)
RDK.RunProgram("Grinder Tool Attach (Tool Stand)", True)
RDK.setSimulationSpeed(0.1)
robot.MoveJ(T_CMF, blocking=True)
robot.MoveL(T_cBut1, blocking=True)
robot.MoveL(T_cBut1App, blocking=True)
RDK.setSimulationSpeed(2)
robot.MoveJ(T_grinderApproach, blocking=True)
RDK.RunProgram("Grinder Tool Attach (Tool Stand)", True)
robot.MoveJ(target, blocking=True)
rdk.pause(1)

# # TASK A
# robot.MoveJ(T_home, blocking=True)
# robot.MoveJ(T_portafilterApproach, blocking=True)
# RDK.RunProgram("Portafilter Tool Attach (Tool Stand)", True)
# RDK.setSimulationSpeed(0.4)
# robot.MoveJ(J_intermediateTF2G, blocking=True)
# robot.MoveJ(T_PF2approach, blocking=True) ### Need more here
# RDK.setSimulationSpeed(0.1)
# robot.MoveL(T_PF2target, blocking=True)
# RDK.setSimulationSpeed(0.2)
# RDK.RunProgram("Portafilter Tool Detach (Tool Stand)", True)
# RDK.setSimulationSpeed(2)
# robot.MoveJ(T_home, blocking=True)

# TASK B
# robot.MoveJ(grinderApproachT, blocking=True)
# RDK.RunProgram("Grinder Tool Attach (Tool Stand)", True)


# RDK.Pause(time_ms=300)

# robot.MoveJ(T_home, blocking=True)

# RDK.RunProgram("Grinder Tool Detach (Tool Stand)", True)
# print(oRs)
