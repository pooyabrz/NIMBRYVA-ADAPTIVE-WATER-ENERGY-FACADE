"""Finite slew and wind sizing for ONE motor+gear reference.

HangTuo SE3 with 24V motor: 600 Nm rated output, 3000 Nm holding,
1500 Nm tilting, .065 rpm output; motor table 30 W and <3 A at 24 V.
30 W is a catalogue nominal proxy, 72 W a running-current envelope, not a
start/stall guarantee. Product table does not qualify a final facade assembly.
"""
import math
import numpy as np
from geometry import panel_frame


def pose(face,tilt):return float(tilt)+180*int(face)


def load_envelope(c,assembly,wind,rho=1.225,angle=0,film_mass=0):
    a=c['area_m2'];chord=c['rotation_chord_m']
    gust=max(0,wind)*c['wind_gust_factor']
    force=.5*rho*gust**2*c['wind_force_coefficient']*a
    # Worst incidence retained for sizing/stowing; no unverified cos^2 relief.
    lever=c['wind_CP_offset_fraction']*chord
    gravity=assembly['moving_mass_kg']*9.80665*c['pivot_CG_offset_m']+film_mass*9.80665*chord/2
    inertia=(assembly['moving_mass_kg']+film_mass)*chord**2/12
    omega=c['motor_output_rpm']*2*math.pi/60
    acceleration=omega/c['motor_ramp_s']
    torque=c['torque_screening_factor']*(force*lever+gravity+c['bearing_friction_Nm']+inertia*acceleration)
    overturn=c['torque_screening_factor']*force*c['wind_mount_lever_m']
    radial=c['torque_screening_factor']*(force+(assembly['moving_mass_kg']+film_mass)*9.80665)
    return {'force_N':force,'required_drive_Nm':torque,'required_hold_Nm':torque,
            'required_tilting_Nm':overturn,'required_radial_N':radial,
            'drive_utilization':torque/c['motor_rated_output_Nm'],
            'hold_utilization':torque/c['motor_holding_Nm'],
            'tilt_utilization':overturn/c['motor_tilting_Nm'],
            'radial_utilization':radial/c['motor_radial_N'],
            'screen_pass':bool(torque<=c['motor_rated_output_Nm'] and overturn<=c['motor_tilting_Nm'] and radial<=c['motor_radial_N'])}


def transition(c,start,end):
    # 0..360 wired excursion, no shortest path through the cable hard limit.
    if not (c['motor_min_output_deg']<=min(start,end) and max(start,end)<=c['motor_max_output_deg']):
        raise ValueError('Requested motion crosses cable/rotation envelope')
    distance=abs(end-start)
    speed=c['motor_output_rpm']*6
    # Trapezoid with prescribed rated speed and a gentle acceleration ramp.
    ramp=c['motor_ramp_s'];tr=min(ramp,math.sqrt(distance*ramp/speed)) if distance else 0
    peak=speed*tr/ramp if ramp else speed
    total=distance/peak+tr if peak else 0
    return {'start':start,'end':end,'distance_deg':distance,'duration_s':total,
            'ramp_s':tr,'peak_deg_s':peak,
            'nominal_Wh':c['motor_running_W']*total/3600,
            'running_upper_Wh':c['motor_running_upper_W']*total/3600}


def motion_angle(move,time):
    duration=move['duration_s'];ramp=move['ramp_s'];speed=move['peak_deg_s']
    t=min(max(time,0),duration)
    if not duration:return move['end']
    if t<ramp:dist=.5*speed*t*t/ramp
    elif t<=duration-ramp:dist=speed*(t-.5*ramp)
    else:dist=move['distance_deg']-.5*speed*(duration-t)**2/ramp
    return move['start']+math.copysign(dist,move['end']-move['start'])


def allowable_wind(c,assembly,ceiling=80):
    from scipy.optimize import brentq
    def margin(u):
        r=load_envelope(c,assembly,u)
        return max(r['drive_utilization'],r['tilt_utilization'],r['radial_utilization'])-1
    return brentq(margin,0,ceiling) if margin(ceiling)>0 else ceiling
