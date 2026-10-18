"""Shared coordinate convention for radiation and the LOCAL Python viewer.

x: facade lateral, y: up, z: outward. Tilt is measured from horizontal.
Ray factors are a deterministic, cosine-weighted reference at the panel centre;
not a mesh-resolved radiation calculation or an aerodynamic model.
"""
from functools import lru_cache
import math
import numpy as np


def panel_frame(tilt_deg):
    b = math.radians(float(tilt_deg))
    return np.array([1., 0., 0.]), np.array([0., math.sin(b), -math.cos(b)]), np.array([0., math.cos(b), math.sin(b)])


def panel_corners(tilt_deg, area_m2=1., height_m=10., offset_m=0.):
    u, v, n = panel_frame(tilt_deg)
    centre = np.array([0., height_m, 0.]) + offset_m*n
    half = math.sqrt(area_m2)/2
    return np.array([centre+half*(a*u+b*v) for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]])


def ray_category(directions, height_m, standoff_m, wall_height_m, wall_width_m):
    """First intersection: 0=sky, 1=ground, 2=finite wall."""
    d = np.atleast_2d(directions)
    tg = np.full(len(d), np.inf)
    down = d[:,1] < -1e-12
    tg[down] = -height_m/d[down,1]
    tw = np.full(len(d), np.inf)
    back = d[:,2] < -1e-12
    tw[back] = -standoff_m/d[back,2]
    x = tw[back]*d[back,0]
    y = height_m+tw[back]*d[back,1]
    valid = (np.abs(x)<=wall_width_m/2)&(y>=0)&(y<=wall_height_m)
    idx = np.flatnonzero(back)
    tw[idx[~valid]] = np.inf
    kind = np.zeros(len(d), dtype=int)
    kind[np.isfinite(tg)] = 1
    kind[(tw<tg)&np.isfinite(tw)] = 2
    return kind


@lru_cache(maxsize=128)
def view_factors(tilt_deg, height_m=10., standoff_m=.7, wall_height_m=30., wall_width_m=20., rays=16384):
    u, v, n = panel_frame(tilt_deg)
    # Deterministic equal-area disc sampling -> cosine-weighted hemisphere.
    k = np.arange(rays)+.5
    radius = np.sqrt(k/rays)
    phi = k*math.pi*(3-math.sqrt(5))
    directions = (radius*np.cos(phi))[:,None]*u + (radius*np.sin(phi))[:,None]*v + np.sqrt(1-radius**2)[:,None]*n
    kind = ray_category(directions, height_m, standoff_m, wall_height_m, wall_width_m)
    return tuple(float(np.mean(kind==i)) for i in range(3))


def solar_vector(zenith_deg, azimuth_deg, facade_azimuth_deg):
    z = np.radians(zenith_deg)
    a = np.radians(np.asarray(azimuth_deg)-facade_azimuth_deg)
    return np.column_stack((np.sin(z)*np.sin(a),np.cos(z),np.sin(z)*np.cos(a)))


def rectangle_corners(angle_deg,width,chord,height,offset=0):
    u,v,n=panel_frame(angle_deg);centre=np.array([0.,height,0.])+offset*n
    return np.array([centre+sx*width/2*u+sy*chord/2*v for sx,sy in [(-1,-1),(1,-1),(1,1),(-1,1)]])


def tiled_corners(angle,height,offset,rows=16,columns=10,side=.1,gap=.01):
    u,v,_=panel_frame(angle);quads=[]
    for row in range(rows):
        for col in range(columns):
            shift=(col-(columns-1)/2)*(side+gap)*u+(row-(rows-1)/2)*(side+gap)*v
            quads.append(rectangle_corners(angle,side,side,height,offset)+shift)
    return quads


def ray_category_from(directions,origin,standoff,wall_height,wall_width):
    d=np.atleast_2d(directions);o=np.asarray(origin)
    tg=np.full(len(d),np.inf);down=d[:,1]<-1e-12
    tg[down]=-o[1]/d[down,1]
    tw=np.full(len(d),np.inf);back=d[:,2]<-1e-12
    tw[back]=(-standoff-o[2])/d[back,2]
    idx=np.flatnonzero(back);x=o[0]+tw[back]*d[back,0];y=o[1]+tw[back]*d[back,1]
    valid=(np.abs(x)<=wall_width/2)&(y>=0)&(y<=wall_height)&(tw[back]>0)
    tw[idx[~valid]]=np.inf
    kind=np.zeros(len(d),dtype=int);kind[np.isfinite(tg)&(tg>0)]=1
    kind[(tw<tg)&np.isfinite(tw)]=2
    return kind


@lru_cache(maxsize=8)
def cassette_view_table(width,chord,height,standoff,wall_height,wall_width,depth,
                        step_deg=2.,rays=1024,area_samples=3):
    """Area quadrature and cosine rays for BOTH physical exterior surfaces.

    Linear interpolation is used during finite motion. No neighbour modules,
    reflective shield, ribs, multiple reflections or CFD are implied.
    """
    angles=np.linspace(0,360,int(round(360/step_deg))+1);table=[]
    k=np.arange(rays)+.5;r=np.sqrt(k/rays);phi=k*math.pi*(3-math.sqrt(5))
    s=(np.arange(area_samples)+.5)/area_samples-.5
    for beta in angles:
        u,v,n=panel_frame(beta);factors=[]
        for sign in (1,-1):
            directions=(r*np.cos(phi))[:,None]*u+(r*np.sin(phi))[:,None]*v+np.sqrt(1-r*r)[:,None]*(sign*n)
            accum=np.zeros(3)
            for x in s:
                for y in s:
                    origin=np.array([0.,height,0.])+x*width*u+y*chord*v+sign*depth/2*n
                    kind=ray_category_from(directions,origin,standoff,wall_height,wall_width)
                    accum+=np.bincount(kind,minlength=3)/rays
            factors.append(accum/(area_samples**2))
        table.append(factors)
    return angles,np.asarray(table)
