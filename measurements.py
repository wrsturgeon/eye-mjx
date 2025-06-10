#!/usr/bin/env python3


import units


N_LEGS = 3


LENGTH_CENTER_TO_ROLL = units.inches(2)
LENGTH_ROLL_TO_PITCH = units.inches(0.5)
LENGTH_PITCH_TO_KNEE = units.inches(2)
LENGTH_KNEE_TO_FOOT = units.inches(4.5)


TOTAL_MASS = units.grams(250)
LEG_DENSITY = units.grams(1) / units.inches(1)
SPHERE_SCAFFOLDING_MASS = units.grams(100)
SERVO_MASS = units.grams(19)
FOOT_CAP_MASS = units.grams(10)


ROLL_ANGLE_MIN = units.degrees(-30)
ROLL_ANGLE_MAX = units.degrees(30)
PITCH_ANGLE_MIN = units.degrees(-90)
PITCH_ANGLE_MAX = units.degrees(90)
KNEE_ANGLE_MIN = units.degrees(-45)
KNEE_ANGLE_MAX = units.degrees(45)


SERVO_TORQUE = units.kg_cm(4)
SERVO_KP = 21.1  # from <https://github.com/google-deepmind/mujoco/issues/1075>: see line <https://github.com/google-deepmind/mujoco_menagerie/blob/cfd91c5605e90f0b77860ae2278ff107366acc87/robotis_op3/op3.xml#L62>


SPHERE_RADIUS = units.inches(5) / 2
LEG_RADIUS = units.inches(0.2)
FOOT_RADIUS = units.inches(0.25)
FLOOR_THICKNESS = units.inches(2)
