#!/usr/bin/env python3


# Using the MKS system of units:
# meters, kilograms, & seconds.


from math import cos, sin
import measurements
from os import makedirs
import units
import xml.etree.ElementTree as XML


# The idea is that you have to import this file,
# which has the side effect of regenerating the XML files,
# to find the path at which the XML files were generated:
MJCF_XML_PATH_ROBOT = "./mjcf/robot.xml"
MJCF_XML_PATH_ENVIRONMENT = "./mjcf/environment.xml"
MJCF_XML_PATH_SCENE = "./mjcf/scene.xml"


ENABLE_SERVOS = True
ENABLE_IMU = True
ENABLE_FSRS = False  # True
ENABLE_SPHERE = True


SIMULATION_TIMESTEP = 0.004


FOOT_FRICTION_SLIDING = 1
FOOT_FRICTION_TORSIONAL = 1
FOOT_FRICTION_ROLLING = 1


JOINT_ARMATURE = 0.005
JOINT_FRICTION_LOSS = 0.1


makedirs("mjcf", exist_ok=True)


################################################################################################################################


robot = XML.Element("mujoco", model="eye_robot")


# Compile-time options:
XML.SubElement(robot, "compiler", autolimits="true")


# Runtime options:
option = XML.SubElement(robot, "option", timestep=f"{SIMULATION_TIMESTEP}")
XML.SubElement(option, "flag", eulerdamp="disable")


# Defaults:
default = XML.SubElement(
    XML.SubElement(robot, "default"), "default", **{"class": "eye"}
)

# Defaults for all `geom`s (geometric physical objects):
XML.SubElement(
    default,
    "geom",
    type="capsule",
    size=f"{measurements.LEG_RADIUS}",
    contype="0",
    conaffinity="0",
    rgba="1 0 0 1",
)

# Defaults for all colliding elements:
XML.SubElement(
    XML.SubElement(default, "default", **{"class": "eye/colliding"}),
    "geom",
    contype="1",
    conaffinity="1",
    rgba="0 0 0 1",
    friction=f"{FOOT_FRICTION_SLIDING} {FOOT_FRICTION_TORSIONAL} {FOOT_FRICTION_ROLLING}",
)

# Defaults for joints:
XML.SubElement(
    default,
    "joint",
    armature=f"{JOINT_ARMATURE}",
    frictionloss=f"{JOINT_FRICTION_LOSS}",
)

# Defaults for servos:
XML.SubElement(
    default,
    "position",
    inheritrange="1",
    kp=f"{measurements.SERVO_KP}",
    forcerange=f"{-measurements.SERVO_TORQUE} {measurements.SERVO_TORQUE}",
    dampratio="1",
)


# Root of all physical objects:
worldbody = XML.SubElement(robot, "worldbody")

# Root of the robot, as opposed to its surroundings:
body = XML.SubElement(
    worldbody,
    "body",
    name="robot",
    childclass="eye",
    pos=f"0 0 {measurements.LENGTH_KNEE_TO_FOOT + measurements.FOOT_RADIUS}",
)

# Free joint (i.e. the robot body can move freely, unattached to its surroundings):
XML.SubElement(body, "freejoint", name="untethered")

# Body comprised only of the spherical shell:
sphere = XML.SubElement(
    body,
    "body",
    name="sphere",
)

XML.SubElement(
    sphere,
    "geom",
    name="sphere",
    type="sphere",
    size=f"{measurements.SPHERE_RADIUS}",
    rgba=f"1 1 1 {1.0 * ENABLE_SPHERE}",
)

if ENABLE_IMU:
    XML.SubElement(sphere, "site", name="imu")


leg_replicator = XML.SubElement(
    body,
    "replicate",
    count=f"{measurements.N_LEGS}",
    euler=f"0 0 {360 / measurements.N_LEGS}",
)

leg = XML.SubElement(
    leg_replicator, "body", name="leg", pos=f"{measurements.LENGTH_CENTER_TO_ROLL} 0 0"
)

XML.SubElement(
    leg,
    "joint",
    name="roll",
    axis="1 0 0",
    range=f"{measurements.ROLL_ANGLE_MIN} {measurements.ROLL_ANGLE_MAX}",
)

XML.SubElement(
    leg,
    "geom",
    name="roll_to_pitch",
    fromto=f"0 0 0 {measurements.LENGTH_ROLL_TO_PITCH} 0 0",
)

leg_after_pitch = XML.SubElement(
    leg, "body", name="leg_after_pitch", pos=f"{measurements.LENGTH_ROLL_TO_PITCH} 0 0"
)

XML.SubElement(
    leg_after_pitch,
    "joint",
    name="pitch",
    axis="0 1 0",
    range=f"{measurements.PITCH_ANGLE_MIN} {measurements.PITCH_ANGLE_MAX}",
)

XML.SubElement(
    leg_after_pitch,
    "geom",
    name="pitch_to_knee",
    fromto=f"0 0 0 {measurements.LENGTH_PITCH_TO_KNEE} 0 0",
)

leg_after_knee = XML.SubElement(
    leg_after_pitch,
    "body",
    name="leg_after_knee",
    pos=f"{measurements.LENGTH_PITCH_TO_KNEE} 0 0",
)

XML.SubElement(
    leg_after_knee,
    "joint",
    name="knee",
    axis="0 1 0",
    range=f"{measurements.KNEE_ANGLE_MIN} {measurements.KNEE_ANGLE_MAX}",
)

XML.SubElement(
    leg_after_knee,
    "geom",
    name="knee_to_foot",
    fromto=f"0 0 0 0 0 {-measurements.LENGTH_KNEE_TO_FOOT}",
)

foot = XML.SubElement(
    leg_after_knee,
    "body",
    name="foot",
    pos=f"0 0 {-measurements.LENGTH_KNEE_TO_FOOT}",
    childclass="eye/colliding",
)

XML.SubElement(
    foot,
    "geom",
    name="foot",
    type="sphere",
    size=f"{measurements.FOOT_RADIUS}",
    mass=f"{measurements.FOOT_CAP_MASS}",
    contype="1",
    conaffinity="1",
)

if ENABLE_FSRS:
    XML.SubElement(foot, "site", name="foot_fsr")


if ENABLE_SERVOS:
    actuator = XML.SubElement(robot, "actuator")
    XML.SubElement(actuator, "position", name="roll", joint="roll")
    XML.SubElement(actuator, "position", name="pitch", joint="pitch")
    XML.SubElement(actuator, "position", name="knee", joint="knee")

sensor = XML.SubElement(robot, "sensor")
if ENABLE_IMU:
    XML.SubElement(sensor, "accelerometer", name="accelerometer", site="imu")
    XML.SubElement(sensor, "gyro", name="gyro", site="imu")
if ENABLE_FSRS:
    XML.SubElement(sensor, "touch", name="fsr", site="foot_fsr")


tree = XML.ElementTree(robot)
XML.indent(tree)
tree.write(MJCF_XML_PATH_ROBOT)


################################################################################################################################


environment = XML.Element("mujoco", model="eye_environment")


# Global stats:
XML.SubElement(environment, "statistic", center="0 0 0", extent="0.8")


# Visual options:
visual = XML.SubElement(environment, "visual")

# Headlight from the camera:
XML.SubElement(
    visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0 0 0"
)

# Haze at teh render limit:
XML.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")

# Default camera:
XML.SubElement(
    visual,
    "global",
    # azimuth=f"{160}",
    azimuth=f"{120}",
    elevation=f"{-20}",
)


# Textures & materials:
asset = XML.SubElement(environment, "asset")

# Sky:
XML.SubElement(
    asset,
    "texture",
    type="skybox",
    builtin="gradient",
    rgb1="0.3 0.5 0.7",
    rgb2="0 0 0",
    width="512",
    height="3072",
)

# Ground:
XML.SubElement(
    asset,
    "texture",
    type="2d",
    name="groundplane",
    builtin="checker",
    mark="edge",
    rgb1="0.2 0.3 0.4",
    rgb2="0.1 0.2 0.3",
    markrgb="0.8 0.8 0.8",
    width="300",
    height="300",
)

# Material:
XML.SubElement(
    asset,
    "material",
    name="groundplane",
    texture="groundplane",
    texuniform="true",
    texrepeat="5 5",
    reflectance="0.2",
)


# Extra physical things:
worldbody = XML.SubElement(environment, "worldbody")

# Floor:
XML.SubElement(
    worldbody,
    "geom",
    name="floor",
    size=f"0 0 {measurements.FLOOR_THICKNESS}",
    type="plane",
    material="groundplane",
    contype="1",
    conaffinity="1",
)


tree = XML.ElementTree(environment)
XML.indent(tree)
tree.write(MJCF_XML_PATH_ENVIRONMENT)


################################################################################################################################


scene = XML.Element("mujoco", model="eye_scene")


# Include the robot and its environment:
XML.SubElement(scene, "include", file="robot.xml")
XML.SubElement(scene, "include", file="environment.xml")


tree = XML.ElementTree(scene)
XML.indent(tree)
tree.write(MJCF_XML_PATH_SCENE)
