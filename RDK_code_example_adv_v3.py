# A more advanced example to get you moving with the RoboDK python API
# Note, as there are many solutions for a given pose, sometimes when
# running this, the robot may choose a weird pose that then doesn't allow
# the subsequent motion (due to being near a singularity etc). If this occurs, 
# just manually reset the robot startingposition to somewhere else and try again
# C Pretty, 18 Sept 2019
# version 2
from robodk.robolink import *
import robolink as rl    # RoboDK API
import robodk as rdk     # Robot toolbox
import numpy as np

RDK = rl.Robolink()

robot = RDK.Item('UR5')
world_frame = RDK.Item('UR5 Base')
target = RDK.Item('Home')   # existing target in station
robot.setPoseFrame(world_frame)
robot.setPoseTool(robot.PoseTool())

# Directly use the RDK Matrix object from to hold pose (its an HT)
T_home = rdk.Mat([[     0.000000,     0.000000,     1.000000,   523.370000 ],
     [-1.000000,     0.000000,     0.000000,  -109.000000 ],
     [-0.000000,    -1.000000,     0.000000,   607.850000 ],
      [0.000000,     0.000000,     0.000000,     1.000000 ]])

# Joint angles
J_intermediatepoint = [-151.880896, -97.616411, -59.103383, -112.890980, 90.242082, -161.879346]

# Convert a numpy array into a Mat (e.g.after calculation)
T_grinderapproach_np = np.array([[     0.173648,    -0.984800,    -0.004000,  -502.103741],
    [ -0.984789,    -0.173618,    -0.006928,  -145.353888 ],
    [  0.006128,     0.005142,    -0.999968,   535.250260 ],
    [  0.000000,     0.000000,     0.000000,     1.000000 ]])

# Need to convert numpy array into and RDK matrix
T_grinderapproach = rdk.Mat(T_grinderapproach_np.tolist())

robot.MoveJ(T_home, blocking=True)
robot.MoveJ(J_intermediatepoint, blocking=True)
robot.MoveL(T_grinderapproach, blocking=True)

# Note that you MUST use the following syntax when calling tool routines
RDK.RunProgram("Grinder Tool Attach (Tool Stand)", True)
RDK.RunProgram("Grinder Tool Detach (Tool Stand)", True)


# and... move home to an existing target
robot.MoveJ(target)


