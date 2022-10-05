from operator import matmul
from sys import gettrace
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

def getScraperRotation(oPs, oPqy, oPq2):
    """Returns the rotation matrix for the scraper given 2 orthogonal points in frame"""
    Py = np.transpose(oPqy - oPs)
    Px = np.transpose(oPs - oPq2)
    print(Px)
    # Px[0][0] += 0.27692
    Pz = np.cross(Px, Py)
    Px = (Px/np.linalg.norm(Px))[0]
    Py = (Py/np.linalg.norm(Py))[0]
    Pz = (Pz/np.linalg.norm(Pz))[0]
    # Px = Px[0]
    # Py = Py[0]
    # Pz = Pz[0]
    
    oRs = np.array([[Px[0], Py[0], Pz[0]],
                    [Px[1], Py[1], Pz[1]],
                    [Px[2], Py[2], Pz[2]]])
    
    return oRs



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

R_ToolFlip = rotationZ(ToolTheta+180)
TInv_grinderPushFlip = np.linalg.inv(getTransform(R_ToolFlip, P_grinderPush))

# Coffee Cup holder
P_cupHoldCentre = np.transpose(np.array([[-47, 0, 186]]))
CCToolTheta = ToolTheta
T_RTool = getTransform(R_Tool, P_Nothing)
T_Centre = getTransform(Identity, P_cupHoldCentre)
T_RX = getTransform(rotationZ(90), P_Nothing)
TInv_cupHoldCentre = np.linalg.inv(np.matmul(np.matmul(T_RTool, T_Centre), T_RX))

P_cupOffset = P_cupHoldCentre + np.transpose(np.array([[-80.0, 0.0, 0]]))      #------Cup Offset-------------
T_CentreOffset = getTransform(Identity, P_cupOffset)
TInv_cupHoldCentreOffset = np.linalg.inv(np.matmul(np.matmul(T_RTool, T_CentreOffset), T_RX))

TInv_cupFlip = np.matmul(getTransform(rotationZ(90), P_cupOffset), TInv_cupHoldCentre)
# R_ToolFlip = rotationZ(CCToolTheta-90)
# P_Cup = P_cupHoldCentre + np.transpose(np.array([[0, 0, 0]]))
# T_RToolFlipCUP = getTransform(R_ToolFlip, P_Nothing)
# T_CentreFlipCup = getTransform(Identity, P_Cup)
# TInv_cupHoldCentreFlipCUP = np.linalg.inv(np.matmul(np.matmul(T_RToolFlipCUP, T_CentreFlipCup), T_RX))

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
P_CC = np.transpose(np.array([[-1.5, -600.3, -20.0]]))
# R_X = rotationX(90)
T_CC = getTransform(Identity, P_CC)
oRcc = rotationX(90)
P_csCntre = np.transpose(np.array([[0.0,0.0,180.0]]))    #--------------------------
T_CupCentre = np.matmul(T_CC, getTransform(oRcc, P_csCntre)).tolist()
T_CUPC = np.matmul(T_CupCentre, TInv_cupHoldCentre).tolist()

P_csCntreApp = P_csCntre + np.transpose(np.array([[0, 70, 0]]))
T_CupCentreApp = np.matmul(T_CC, getTransform(oRcc, P_csCntreApp)).tolist()
T_CUPCAPP = np.matmul(T_CupCentreApp, TInv_cupHoldCentre).tolist()

P_csCntreUp = P_csCntre + np.transpose(np.array([[0, 0, 200]]))
T_CupCentreUp = np.matmul(T_CC, getTransform(oRcc, P_csCntreUp)).tolist()
T_CUPCUP = np.matmul(T_CupCentreUp, TInv_cupHoldCentre).tolist()

# T_CC2 = T_CC@getTransform(oRcc, P_csCntre).tolist()
# P_csCuplip = np.transpose(np.array([[-40.0,0.0,180.0]]))
# P_csCntre = np.transpose(np.array([[0.0,0.0,217.0]]))
# P_csCntreApp = np.transpose(np.array([[0.0,0.0,217.0]]))
# T_CupLip = getTransform(oRcc, P_csCuplip)
# T_CupCentre = np.matmul(getTransform(oRcc, P_csCntre), T_RZ).tolist()
# T_CUP1 = np.matmul(T_CC, TInv_cupHoldCentre).tolist()
# T_CUP2 = np.matmul(T_CUP1, TInv_cupHoldCentre).tolist()
# # T_CUPC = T_CC@T_CupCentre@TInv_cupHoldCentre 
# T_CUPC = np.matmul(np.matmul(T_CC, T_CupCentre), TInv_cupHoldCentre).tolist()


#************************************************************************************
# SCRAPER FRAME
#************************************************************************************
oPs = np.transpose(np.array([[599.9, 53.0, 254.4]]))
oPqy = np.transpose(np.array([[677.9, 69.9, 249.8]])) 
oPq2 = np.transpose(np.array([[582.7, 128.6, 235.8]]))
sPq = np.transpose(np.array([[-80.0, 0.0, 0.0]]))
oRs = getScraperRotation(oPs, oPqy, oPq2)
T_S = getTransform(oRs, oPs)
# Tamper
sPTamp = np.transpose(np.array([[-75, -2, -61]]))                        #---------Tamper----
sPTampApproach = sPTamp + np.transpose(np.array([[0, 0, -50]]))         #-------------------
sRy = rotationY(-90)
sTRx = getTransform(rotationX(-90), P_Nothing)
T_Tamp = np.matmul(T_S, np.matmul(getTransform(sRy, sPTamp), sTRx).tolist()).tolist()
T_TAMP = np.matmul(T_Tamp, TInv_PF1Tool).tolist()
T_TampApp = np.matmul(T_S, np.matmul(getTransform(sRy, sPTampApproach), sTRx).tolist()).tolist()
T_TAMPAPP = np.matmul(T_TampApp, TInv_PF1Tool).tolist()
# Scraper
halfScrape = 40                                                     #----length------Scraper---
dropHeight = 15                                                     #----height------Scraper---
sPScrapeEnd = np.transpose(np.array([[70, halfScrape, -32-dropHeight]]))
sPScrapeStart = np.transpose(np.array([[70, -2*halfScrape, -32-dropHeight]]))
sRy = rotationY(-90)
sTRx = getTransform(rotationX(-90), P_Nothing)
T_ScrapeStart = np.matmul(T_S, np.matmul(getTransform(sRy, sPScrapeStart), sTRx).tolist()).tolist()
T_SCRAPESTART = np.matmul(T_ScrapeStart, TInv_PF1Tool).tolist()
T_ScrapeEnd = np.matmul(T_S, np.matmul(getTransform(sRy, sPScrapeEnd), sTRx).tolist()).tolist()
T_SCRAPEEND = np.matmul(T_ScrapeEnd, TInv_PF1Tool).tolist()




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
P_PF2 = np.transpose(np.array([[160, -18.0, -237]]))             #-----PF2--------------
P_PF2App = np.transpose(np.array([[200, -18.0, -237]]))          #----------------------
PF2Theta = -90 - portTheta-2                                     #----------------------
PF2ThetaApp = -90 - portTheta # Reducing angles tilts head low   #----------------------
R_PF2 = rotationY(PF2Theta)
T_PF2 = getTransform(R_PF2, P_PF2)
T_GPORTPLACE = np.matmul(np.matmul(T_G, T_PF2), TInv_PF2Tool).tolist()
R_PF2App = rotationY(PF2Theta)
T_PF2App = getTransform(R_PF2App, P_PF2App)
T_GPORTPLACEAPP = np.matmul(np.matmul(T_G, T_PF2App), TInv_PF2Tool).tolist()
# Slider
P_Slider = np.transpose(np.array([[-45.82, 110, -103.00]]))
sliderTheta = 100
R_Slider = rotationZ(sliderTheta)
T_Slider1 = getTransform(R_Slider, P_Slider)
R_Slider2 = rotationX(-90)
T_Slider2 = getTransform(R_Slider2, np.transpose(np.array([[0, 0, 0]])))
T_Slider = np.matmul(T_Slider1, T_Slider2)
T_SliderS = np.matmul(T_G, T_Slider)
T_SLIDERSTART = np.matmul(T_SliderS, TInv_grinderPull).tolist()
beta = 61
radius = 120
P_SliderEnd = getPforSlider(beta, radius)
R_SliderEnd = rotationY(beta)
T_SLIDEREND = np.matmul(np.matmul(T_SliderS, getTransform(R_SliderEnd, P_SliderEnd)), TInv_grinderPull).tolist()
P_SliderHalf = getPforSlider(beta/2, radius)
R_SliderHalf = rotationY(beta/2)
T_SLIDERHALF = np.matmul(np.matmul(T_SliderS, getTransform(R_SliderHalf, P_SliderHalf)), TInv_grinderPull).tolist()
# Button
P_gButOn = np.transpose(np.array([[-64.42, 89.82, -227.68]]))
P_gButOff = np.transpose(np.array([[-80.71, 94.26, -227.68]]))
R_gBut = rotationX(90)
T_ButOn = np.matmul(T_G, getTransform(R_gBut, P_gButOn)).tolist()
T_ButOff = np.matmul(T_G, getTransform(R_gBut, P_gButOff)).tolist()

T_BUTONAPP = np.matmul(T_ButOn, TInv_grinderPushPlus).tolist()
T_BUTOFFAPP = np.matmul(T_ButOff, TInv_grinderPushPlus).tolist()
T_BUTONTARG = np.matmul(T_ButOn, TInv_grinderPush).tolist()
T_BUTOFFTARG = np.matmul(T_ButOff, TInv_grinderPush).tolist()

#************************************************************************************
# TOOL FRAME
#************************************************************************************
oPt = np.transpose(np.array([[-555.6, -78.5, 19.05]]))
oPq = np.transpose(np.array([[-645.0, 77.2, 19.05]]))
tPq = np.transpose(np.array([[-127.0, 127.0, 0]]))
oRt, thetaT = getRotation(oPt, oPq, tPq)
T_TF = getTransform(oRt, oPt)
TTT = np.matmul(T_TF, TInv_PF1Tool).tolist()
# Group head
GHdrop = 40                                               #-------------------------
oRGHY = rotationY(-90)
T_RYGHead = getTransform(oRGHY, P_Nothing)
P_GHead = np.transpose(np.array([[11,71,190]]))        #-------------------------
PGHdrop = np.transpose(np.array([[0, 0, -GHdrop]]))
P_GHeadApp = P_GHead + PGHdrop
T_GHead = np.matmul(getTransform(Identity, P_GHead), T_RYGHead)
T_GHeadApp = np.matmul(getTransform(Identity, P_GHeadApp), T_RYGHead)
T_GHEAD = np.matmul(np.matmul(T_TF, T_GHead), TInv_PF1Tool).tolist()
T_GHEADAPP = np.matmul(np.matmul(T_TF, T_GHeadApp), TInv_PF1Tool).tolist()
alpha = 45                                               #-------------------------
N = 15                                                   #-------------------------
T_GHEADRlist = []
for n in range(1, (N)):
    angle = alpha-n*(alpha/N)
    R_GHead = rotationX(angle)
    T_RGHead = np.matmul(T_GHead, getTransform(R_GHead, P_Nothing))
    T_GHEADRlist.append(np.matmul(np.matmul(T_TF, T_RGHead), TInv_PF1Tool).tolist())
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
P_cBut1 = np.transpose(np.array([[50,35.25,-55]]))
R_cBut1 = rotationX(-90)
T_cBut1 = getTransform(R_cBut1, P_cBut1)
T_But1 = np.matmul(T_CMF, T_cBut1).tolist()
T_BUT1BOTTOM = np.matmul(T_But1, TInv_grinderPushFlip).tolist()

P_cBut1Top = P_cBut1 + np.transpose(np.array([[0, 0, 40]]))
T_cBut1Top = getTransform(R_cBut1, P_cBut1Top)
T_But1Top = np.matmul(T_CMF, T_cBut1Top).tolist()
T_BUT1TOP = np.matmul(T_But1Top, TInv_grinderPushFlip).tolist()

# Cup Placement
P_CupPlace = np.transpose(np.array([[0, 72, -290]]))               #-------Cup Place--------
T_CupPos = getTransform(rotationX(-90), P_CupPlace)
T_CupPlace = np.matmul(T_CMF, T_CupPos).tolist()
# T_CupAngle = getTransform(rotationX(0), P_Nothing)
# T_CupPlaceAngle = np.matmul(T_CupPlace, T_CupAngle)
T_CUPPLACE = np.matmul(T_CupPlace, TInv_cupHoldCentreOffset).tolist()

P_CupPlaceApp2 = P_CupPlace + np.transpose(np.array([[0, -150, 0]]))   #-------Cup Approaches--------
T_CupPosApp2 = getTransform(rotationX(-90), P_CupPlaceApp2)
T_CupPlaceApproach2 = np.matmul(T_CMF, T_CupPosApp2)
T_CUPPAPP2 = np.matmul(T_CupPlaceApproach2, TInv_cupHoldCentreOffset).tolist()

# P_CupPlaceApp1 = P_CupPlace + np.transpose(np.array([[0, 0, 10]]))      #-------Cup Approaches--------
# T_CupPosApp1 = getTransform(rotationY(-90), P_CupPlaceApp1)
# T_CupPlaceApproach1 = np.matmul(np.matmul(T_CMF, T_CupPosApp1), T_CupAngle)
# T_CUPPAPP1 = np.matmul(T_CupPlaceApproach1, TInv_cupFlip).tolist()


#************************************************************************************
#  JOINT ANGLES
#************************************************************************************
# Grinder
J_SliderToHome = [-249.120000, -89.240000, 69.370000, -47.120000, 78.200000, -128.000000]
J_toSlider2 = [-257.160000, -78.850000, 44.820000, -53.390000, 92.010000, -147.450000] 
J_toSlider3 = [-253.670000, -74.430000, 104.810000, -34.940000, 92.660000, -141.270000]
J_toPortaDrop1 = [-17.800000, -91.210000, -145.440000, -117.600000, -63.600000, -222.730000]
J_toPortaDrop2 = [-75.530000, -79.930000, -101.290000, -114.160000, -3.150000, -198.260000]
J_Gbut1= [-74.790000, -116.970000, -53.070000, -103.090000, 174.720000, -205.420000]
J_Gbut2 = [-63.030000, -135.630000, -77.200000, -102.170000, 182.700000, -209.000000]

# Scraper
J_GtoScraper = [-17.800000, -89.800000, -146.440000, -115.600000, -63.600000, -222.730000]
J_ScrapeToTamp = [15.390000, -81.780000, -137.880000, -142.220000, -101.540000, -232.610000]

# Coffee Machine
J_CtoBut1 = [4.300000, -97.240000, 115.630000, -209.500000, 54.830000, -107.990000]
J_CtoBut2 = [4.310000, -94.240000, 118.630000, -162.500000, 54.830000, -107.990000]

# Tool Frame
J_ToolFrame = [-149.664712, -68.684232, -83.849623, -116.347560, 91.413927, -0.018507]
J_ToolFrameGrinder = [-140.660000, -89.680000, -75.840000, -100.340000, 90.410000, -0.010000]
# Cup Tool
J_CupApproach = [-298.400000, -91.880000, -204.550000, -68.350000, 62.280000, -31.900000]
J_CupApproach1 = [-62.280000, -61.920000, -144.350000, -153.720000, -65.310000, -41.010000]

J_CupExit = [-64.430000, -96.700000, -127.140000, -283.140000, -11.060000, -65.000000]
J_CupExit1 = [-83.540000, -62.100000, -132.440000, -171.690000, -65.260000, -41.060000]

# Group Head
J_toGroup = [-99.460000, -65.980000, -152.530000, -140.340000, -61.170000, -221.800000]
J_GroupExit = [-167.870000, -65.980000, -152.530000, -140.340000, -61.170000, -221.800000]



### MOVEMENTS
RDK.setSimulationSpeed(0.2)
# # # Place portafilter tool under grinder
# # RDK.RunProgram("Grinder Tool Detach (Tool Stand)", True)

# robot.MoveJ(J_ToolFrame, blocking=True)
# RDK.RunProgram("Portafilter Tool Attach (Tool Stand)", True)
# robot.MoveJ(J_toPortaDrop2, blocking=True)
# robot.MoveJ(J_toPortaDrop1, blocking=True)
# robot.MoveJ(rdk.Mat(T_GPORTPLACEAPP), blocking=True)
# robot.MoveJ(rdk.Mat(T_GPORTPLACE), blocking=True)
# RDK.RunProgram("Portafilter Tool Detach (Grinder)", True)
# robot.MoveJ(rdk.Mat(T_GPORTPLACEAPP), blocking=True)
# robot.MoveJ(J_toPortaDrop1, blocking=True)

# # # Push Grinder Button #DONE
#robot.MoveJ(rdk.Mat(J_ToolFrame), blocking=True)
# # # RDK.RunProgram("Portafilter Tool Detach (Tool Stand)", True)
# robot.MoveJ(J_ToolFrame, blocking=True)
# RDK.RunProgram("Grinder Tool Attach (Tool Stand)", True)
# robot.MoveJ(rdk.Mat(J_Gbut1), blocking=True)
# robot.MoveJ(rdk.Mat(J_Gbut2), blocking=True)
# robot.MoveJ(rdk.Mat(T_BUTONAPP), blocking=True)
# robot.MoveL(rdk.Mat(T_BUTONTARG), blocking=True)
# robot.MoveL(rdk.Mat(T_BUTONAPP), blocking=True)
# robot.pause(3000)
# robot.MoveJ(rdk.Mat(T_BUTOFFAPP), blocking=True)
# robot.MoveL(rdk.Mat(T_BUTOFFTARG), blocking=True)
# robot.MoveL(rdk.Mat(T_BUTOFFAPP), blocking=True)
# robot.MoveJ(rdk.Mat(J_Gbut2), blocking=True)
# robot.MoveJ(rdk.Mat(J_Gbut1), blocking=True)

# # Pull slider #DONE
# robot.MoveJ(rdk.Mat(T_SLIDERSTART), blocking=True)
# robot.MoveC(rdk.Mat(T_SLIDERHALF), rdk.Mat(T_SLIDEREND), blocking=True)
# robot.MoveC(rdk.Mat(T_SLIDERHALF), rdk.Mat(T_SLIDERSTART), blocking=True)
# robot.MoveC(rdk.Mat(T_SLIDERHALF), rdk.Mat(T_SLIDEREND), blocking=True)
# robot.MoveC(rdk.Mat(T_SLIDERHALF), rdk.Mat(T_SLIDERSTART), blocking=True)
# robot.MoveC(rdk.Mat(T_SLIDERHALF), rdk.Mat(T_SLIDEREND), blocking=True)
# robot.MoveC(rdk.Mat(T_SLIDERHALF), rdk.Mat(T_SLIDERSTART), blocking=True)
# robot.MoveJ(rdk.Mat(J_Gbut1), blocking=True)
# robot.MoveJ(J_ToolFrame, blocking=True)
# RDK.RunProgram("Grinder Tool Detach (Tool Stand)", True)

# # Pick up portafilter from PF2
# robot.MoveJ(J_toPortaDrop2, blocking=True)
# robot.MoveJ(rdk.Mat(T_GPORTPLACEAPP), blocking=True)
# RDK.RunProgram("Portafilter Tool Attach (Grinder)", True)
# robot.MoveJ(rdk.Mat(T_GPORTPLACEAPP), blocking=True)
# robot.MoveJ(J_GtoScraper, blocking=True)

# # Scraper and Tamp
# # Scrape
# robot.MoveJ(target, blocking=True)        ---------------
# robot.MoveJ(rdk.Mat(T_SCRAPESTART), blocking=True)
# robot.MoveL(rdk.Mat(T_SCRAPEEND), blocking=True)
# robot.MoveJ(rdk.Mat(T_SCRAPESTART), blocking=True)
# robot.MoveJ(rdk.Mat(J_ScrapeToTamp), blocking=True)
# # Tamp
# robot.MoveJ(rdk.Mat(T_TAMPAPP), blocking=True)
# robot.MoveL(rdk.Mat(T_TAMP), blocking=True)
# robot.MoveL(rdk.Mat(T_TAMPAPP), blocking=True)
# robot.MoveJ(rdk.Mat(J_ScrapeToTamp), blocking=True)
#RDK.setSimulationSpeed(1)                   #---------------
#robot.MoveJ(target, blocking=True)

# Group Head
# robot.MoveJ(rdk.Mat(J_toGroup), blocking=True)
# robot.MoveJ(rdk.Mat(T_GHEADAPP), blocking=True)
# robot.MoveL(rdk.Mat(T_GHEAD), blocking=True)
# for T_GHEADR in reversed(T_GHEADRlist):
#     robot.MoveL(rdk.Mat(T_GHEADR), blocking=True)
# for T_GHEADR in T_GHEADRlist:
#     robot.MoveL(rdk.Mat(T_GHEADR), blocking=True)
# robot.MoveL(rdk.Mat(T_GHEAD), blocking=True)
# robot.MoveL(rdk.Mat(T_GHEAD), blocking=True)
# robot.MoveL(rdk.Mat(T_GHEADAPP), blocking=True)
# robot.MoveJ(rdk.Mat(J_GroupExit), blocking=True)

# # Get Cup
# robot.MoveJ(rdk.Mat(J_ToolFrame), blocking=True)
# RDK.RunProgram("Cup Tool Attach (Stand)", True)
# robot.MoveJ(J_CupApproach1, blocking=True)
# robot.MoveJ(rdk.Mat(T_CUPCAPP), blocking=True)
# RDK.RunProgram("Cup Tool Open", True)
# robot.MoveJ(rdk.Mat(T_CUPC), blocking=True)
# RDK.RunProgram("Cup Tool Close", True)
# robot.MoveL(rdk.Mat(T_CUPCUP), blocking=True)
# robot.MoveJ(J_CupExit1, blocking=True)

# # Place Cup at machine
# robot.MoveJ(J_CupExit, blocking=True)
# robot.MoveJ(rdk.Mat(T_CUPPAPP2), blocking=True)
# robot.MoveJ(rdk.Mat(T_CUPPLACE), blocking=True)
# RDK.RunProgram("Cup Tool Open", True)
# robot.MoveJ(rdk.Mat(T_CUPPAPP2), blocking=True)
# RDK.RunProgram("Cup Tool Close", True)


# # # Push button on coffee machine
# robot.MoveJ(rdk.Mat(J_ToolFrame), blocking=True)
# RDK.RunProgram("Cup Tool Detach (Stand)", True)
# RDK.RunProgram("Grinder Tool Attach (Tool Stand)", True)
# robot.MoveL(rdk.Mat(T_BUT1TOP), blocking=True)
# # Pause
# robot.MoveJ(rdk.Mat(T_BUT1BOTTOM), blocking=True)

# Get Cup of Coffee on Stand



