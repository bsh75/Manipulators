from operator import matmul
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
    aPbNorm = aPb[0:2]/np.linalg.norm(aPb[0:2])
    # print(aPbNorm)
    alpha = math.acos(aPbNorm[1])
    if aPbNorm[0] < 0:
        alpha = -alpha
    v = oPb - oPa
    vNorm = v[0:2]/np.linalg.norm(v[0:2])
    # print(vNorm, '\n', vNorm[1])
    beta = math.acos(vNorm[1])
    if vNorm[0] < 0:
        beta = -beta
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

def rotationY(angle):
    """Returns a rotation matrix about Y axis"""
    angle = angle*np.pi/180
    return np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])

def rotationZ(angle):
    """Returns a rotation matrix about Z axis"""
    angle = angle*np.pi/180
    return np.array([[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]])

def rotationX(angle):
    """Returns a rotation matrix about X axis"""
    angle = angle*np.pi/180
    return np.array([[1, 0.0, 0.0], [0.0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]])

def getPforSlider(angle, radius):
    """return a p vector for a given rotation about Y 
    and a given length away from center of rotation"""
    angle = angle * np.pi/180
    x = 2*radius*sin(angle/2)
    alpha = np.pi/2 - (np.pi/2 - angle/2)
    P = np.transpose(np.array([[-x*math.sin(alpha), 0, -x*math.cos(alpha)]]))
    return P


#************************************************************************************
# TOOLS
#************************************************************************************
Identity = np.array([[1, 0, 0], [0, 1, 0],[0, 0, 1]])
R_FlipZ = np.array([[1, 0, 0], [0, -1, 0],[0, 0, -1]])
Identity = np.array([[1, 0, 0], [0, 1, 0],[0, 0, 1]])
ToolTheta = -50
R_Tool = rotationZ(ToolTheta)
# Portafilter
portTheta = -7.5
R_porta = rotationY(portTheta)
P_PF2Tool = np.transpose(np.array([[-32.0, 0, 27.56]]))
TInv_PF2Tool = np.linalg.inv(getTransform(R_Tool, P_PF2Tool))
P_PF1Tool = np.transpose(np.array([[4.71, 0, 144.76]]))
P_Nothing = np.transpose(np.array([[0, 0, 0]]))
TInv_PF1Tool = np.linalg.inv(np.matmul(getTransform(R_Tool, P_PF1Tool), getTransform(R_porta, P_Nothing)))
# Grinder
P_grinderPush = np.transpose(np.array([[0.0, 0.0, 102.82]]))
P_grinderPushPlus = np.transpose(np.array([[0.0, 0.0, 120.82]]))
TInv_grinderPush = np.linalg.inv(getTransform(R_Tool, P_grinderPush))
TInv_grinderPushPlus = np.linalg.inv(getTransform(R_Tool, P_grinderPushPlus))
P_grinderPull = np.transpose(np.array([[-50.0, 0, 67.06]]))
TInv_grinderPull = np.linalg.inv(getTransform(R_Tool, P_grinderPull))
# Coffee Cup holder
P_cupHoldCentre = np.transpose(np.array([[-47.0, 0.0, 186.11]]))
TInv_cupHoldCentre = np.linalg.inv(getTransform(R_Tool, P_cupHoldCentre))

# tcpTgt = np.linalg.inv(getTransform(R_grinder, np.transpose(np.array([[0, 0, 0, 0]]))))
# gtTpb = np.linalg.inv(getTransform(Identity, P_grinderPush))
# tcpTpb = np.matmul(gtTpb, tcpTgt)
# print(T_grinderPush)
# print(tcpTpb)
# print(T_grinderPush)
# print("T_grinderPush = ", T_grinderPush)

#************************************************************************************
# COFFEE CUP FRAME
#************************************************************************************
P_CC = np.transpose(np.array([[-1.1, -600.8, -20.0]]))
T_CC = getTransform(Identity, P_CC)
P_csCuplip = np.transpose(np.array([[-40.0,0.0,180.0]]))
P_csCntre = np.transpose(np.array([[0.0,0.0,217.0]]))
T_CupLip = getTransform(Identity, P_csCuplip)
T_CupCentre = np.matmul(getTransform(Identity, P_csCntre), TInv_cupHoldCentre).tolist()


#************************************************************************************
# SCRAPER FRAME
#************************************************************************************
oPs = np.transpose(np.array([[599.9, 53.0, 254.4]]))
oPq = np.transpose(np.array([[582.7, 128.6, 235.8]]))
oPq2 = np.transpose(np.array([[677.9, 69.9, 249.8]])) 
sPq = np.transpose(np.array([[-80.0, 0.0, 0.0]]))
P_sTamper = np.transpose(np.array([[-80.0, 0.0, -55.0]]))
P_sScraper = np.transpose(np.array([[70.0,0.0,-32.0]]))
oRs, thetaS = getRotation(oPs, oPq, sPq)
print(thetaS)
# oRs, thetaS = getRotation(oPs, oPq2, sPq)
# print(thetaS)

#************************************************************************************
# GRINDER FRAME ###
#************************************************************************************
oPg = np.transpose(np.array([[482.7, -432.1, 316.1]]))
oPqg = np.transpose(np.array([[370.5, -322.5, 65.9]]))
gPq = np.transpose(np.array([[157.61, 0.0, -250.45]]))
oRg, thetaG = getRotation(oPg, oPqg, gPq)
# oRg = rotationZ(135)
T_G = getTransform(oRg, oPg)
# PF2
P_PF2 = np.transpose(np.array([[157.61, -20.0, -240]]))
P_PF2App = np.transpose(np.array([[200, -20.0, -230]]))
PF2Theta = -90 - portTheta +4
PF2ThetaApp = -90 - portTheta +4
R_PF2 = rotationY(PF2Theta)
T_PF2 = getTransform(R_PF2, P_PF2)
T_GportPlace = np.matmul(np.matmul(T_G, T_PF2), TInv_PF2Tool).tolist()
R_PF2App = rotationY(PF2Theta)
T_PF2App = getTransform(R_PF2App, P_PF2App)
T_GportPlaceApp = np.matmul(np.matmul(T_G, T_PF2App), TInv_PF2Tool).tolist()
# Slider
P_Slider = np.transpose(np.array([[-45.82, 110, -103.00]]))
sliderTheta = 100
R_Slider = rotationZ(sliderTheta)
T_Slider1 = getTransform(R_Slider, P_Slider)
R_Slider2 = rotationX(-90)
T_Slider2 = getTransform(R_Slider2, np.transpose(np.array([[0, 0, 0]])))
T_Slider = np.matmul(T_Slider1, T_Slider2)
T_SliderS = np.matmul(T_G, T_Slider)
T_SliderStart = np.matmul(T_SliderS, TInv_grinderPull).tolist()
beta = 61
radius = 120
P_SliderEnd = getPforSlider(beta, radius)
R_SliderEnd = rotationY(beta)
T_SliderEnd = np.matmul(np.matmul(T_SliderS, getTransform(R_SliderEnd, P_SliderEnd)), TInv_grinderPull).tolist()
P_SliderHalf = getPforSlider(beta/2, radius)
R_SliderHalf = rotationY(beta/2)
T_SliderHalf = np.matmul(np.matmul(T_SliderS, getTransform(R_SliderHalf, P_SliderHalf)), TInv_grinderPull).tolist()
# Button
P_gButOn = np.transpose(np.array([[-64.42, 89.82, -227.68]]))
P_gButOff = np.transpose(np.array([[-80.71, 94.26, -227.68]]))
R_gBut = rotationX(90)
T_ButOn = np.matmul(T_G, getTransform(R_gBut, P_gButOn)).tolist()
T_ButOff = np.matmul(T_G, getTransform(R_gBut, P_gButOff)).tolist()

T_ButOnApproach = np.matmul(T_ButOn, TInv_grinderPushPlus).tolist()
T_ButOffApproach = np.matmul(T_ButOff, TInv_grinderPushPlus).tolist()
T_ButOnTarg = np.matmul(T_ButOn, TInv_grinderPush).tolist()
T_ButOffTarg = np.matmul(T_ButOff, TInv_grinderPush).tolist()

#************************************************************************************
# TOOL FRAME
#************************************************************************************
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
T_TF = getTransform(oRt, oPt)
T_gTool = getTransform(R_FlipZ, gtaP)
T_gToolTarg = np.matmul(T_TF, T_gTool).tolist()
T_pfTool = getTransform(R_FlipZ, ptaP)
T_pfToolTarg = np.matmul(T_TF, T_pfTool).tolist()
T_cTool = getTransform(R_FlipZ, ctaP)
T_cToolTarg = np.matmul(T_TF, T_pfTool).tolist()
# print(T_grinderApproach)
# T_portafilterApproach = rdk.Mat(getTransform(oRt, ptaP))
# T_cupApproach = rdk.Mat(getTransform(oRt, ctaP))
# T_portafilterEntryApproach = rdk.Mat(getTransform(oRt, portafilterEntryP))
# print(oPt + np.matmul(oRt, tPq), "==\n", oPq, '\n') #GOOD

#************************************************************************************
# COFFEE MACHINE FRAME
#************************************************************************************
oPc = np.transpose(np.array([[-365.5, -386.7, 349.8]]))
oPq = np.transpose(np.array([[-577.8, -441.6, 349.8]]))
cPq = np.transpose(np.array([[0.0, 218.0, 0.0]]))
oRc, thetaC = getRotation(oPc, oPq, cPq)
T_CMF = getTransform(oRc, oPc)
T_CMF2 = np.matmul(T_CMF, TInv_grinderPush).tolist()
# print(T_CMF)
# Button 1: urTtcp = urTcm cmTbut1 inv(gtTpb) inv(tcpTgt)
P_cBut1 = np.transpose(np.array([[45.67,35.25,-45.89]]))
R_cBut1 = rotationY(-90)
T_cBut1 = getTransform(R_cBut1, P_cBut1)
T_But1 = np.matmul(T_CMF, T_cBut1)
T_But1Approach = np.matmul(T_But1, TInv_grinderPushPlus).tolist()
T_But1Target = np.matmul(T_But1, TInv_grinderPush).tolist()
P_cBut2 = np.transpose(np.array([[50.67,35.25,-61.39]]))
P_cBut3 = np.transpose(np.array([[50.67,35.25,-94.89]]))
P_cSwtch1 = np.transpose(np.array([[50.67,98.75, -27.89]]))
P_cCupSpot= np.transpose(np.array([[-12.68,72.0,-290]]))

#************************************************************************************
#  JOINT ANGLES
#************************************************************************************
# Grinder
J_SliderToHome = [-249.120000, -89.240000, 69.370000, -47.120000, 78.200000, -128.000000]
J_toSlider2 = [-257.160000, -78.850000, 44.820000, -53.390000, 92.010000, -147.450000] 
J_toSlider3 = [-253.670000, -74.430000, 104.810000, -34.940000, 92.660000, -141.270000]
J_toPortaDrop1 = [-238.380000, -105.180000, 156.020000, -71.390000, 75.960000, -219.120000]
J_toPortaDrop2 = [-238.380000, -95.180000, 146.020000, -52.390000, 75.960000, -219.120000]
J_Gbut1= [-74.790000, -116.970000, -53.070000, -103.090000, 174.720000, -205.420000]
J_Gbut2 = [-63.030000, -135.630000, -77.200000, -102.170000, 182.700000, -209.000000]

# Coffee Machine
J_CtoBut1 = [4.300000, -97.240000, 115.630000, -209.500000, 54.830000, -107.990000]
J_CtoBut2 = [4.310000, -94.240000, 118.630000, -162.500000, 54.830000, -107.990000]

# Tool Frame
J_ToolFrame = [-149.664712, -68.684232, -83.849623, -116.347560, 91.413927, -0.018507]


### MOVEMENTS

# # Place portafilter tool under grinder
# RDK.RunProgram("Grinder Tool Detach (Tool Stand)", True)
#robot.MoveJ(J_ToolFrame, blocking=True)
#RDK.RunProgram("Portafilter Tool Attach (Tool Stand)", True)
#robot.MoveJ(J_toPortaDrop1, blocking=True)
robot.MoveJ(rdk.Mat(T_GportPlaceApp), blocking=True)
robot.MoveJ(rdk.Mat(T_GportPlace), blocking=True)
RDK.RunProgram("Portafilter Tool Detach (Grinder)", True)
robot.MoveJ(rdk.Mat(T_GportPlace), blocking=True)
robot.MoveJ(rdk.Mat(T_GportPlaceApp), blocking=True)
robot.MoveJ(J_toPortaDrop1, blocking=True)
robot.MoveJ(target, blocking=True)

# # # Push Grinder Button
#robot.MoveJ(rdk.Mat(J_ToolFrame), blocking=True)
# # # RDK.RunProgram("Portafilter Tool Detach (Tool Stand)", True)
#RDK.RunProgram("Grinder Tool Attach (Tool Stand)", True)
#robot.MoveJ(rdk.Mat(J_Gbut1), blocking=True)
#robot.MoveJ(rdk.Mat(J_Gbut2), blocking=True)
#robot.MoveJ(rdk.Mat(T_ButOnApproach), blocking=True)
#robot.MoveL(rdk.Mat(T_ButOnTarg), blocking=True)
#robot.MoveL(rdk.Mat(T_ButOnApproach), blocking=True)
#robot.MoveJ(rdk.Mat(T_ButOffApproach), blocking=True)
#robot.MoveL(rdk.Mat(T_ButOffTarg), blocking=True)
#robot.MoveL(rdk.Mat(T_ButOffApproach), blocking=True)
#robot.MoveJ(rdk.Mat(J_Gbut2), blocking=True)
#robot.MoveJ(rdk.Mat(J_Gbut1), blocking=True)
#robot.MoveJ(target, blocking=True)

# # Pull slider 
#robot.MoveJ(rdk.Mat(J_SliderToHome), blocking=True)
#robot.MoveJ(rdk.Mat(J_toSlider2), blocking=True)
#robot.MoveJ(rdk.Mat(J_toSlider3), blocking=True)
#robot.MoveJ(rdk.Mat(T_SliderStart), blocking=True)
#robot.MoveC(rdk.Mat(T_SliderHalf), rdk.Mat(T_SliderEnd), blocking=True) #MoveC(target1, target2, itemrobot, blocking=True)
#robot.MoveC(rdk.Mat(T_SliderHalf), rdk.Mat(T_SliderStart), blocking=True) #MoveC(target1, target2, itemrobot, blocking=True)
#robot.MoveJ(rdk.Mat(J_SliderToHome), blocking=True)
#robot.MoveJ(target, blocking=True)


# # # Push button on coffee machine
# # robot.MoveJ(rdk.Mat(T_gToolTarg), blocking=True)
# # RDK.RunProgram("Grinder Tool Attach (Tool Stand)", True)
#robot.MoveJ(J_CtoBut2, blocking=True)
#robot.MoveJ(rdk.Mat(T_But1Approach), blocking=True)
#robot.MoveL(rdk.Mat(T_But1Target), blocking=True)
#robot.MoveL(rdk.Mat(T_But1Approach), blocking=True)
#robot.MoveJ(J_CtoBut1, blocking=True)
#robot.MoveJ(rdk.Mat(T_gToolTarg), blocking=True)
# # RDK.RunProgram("Grinder Tool Detach (Tool Stand)", True)
# robot.MoveJ(target, blocking=True)

# # Get Cup
# robot.MoveJ(rdk.Mat(T_cToolTarg), blocking=True)
# RDK.RunProgram("Cup Tool Attach (Stand)", True)
# robot.MoveJ(rdk.Mat(T_CupCentre, blocking=True))