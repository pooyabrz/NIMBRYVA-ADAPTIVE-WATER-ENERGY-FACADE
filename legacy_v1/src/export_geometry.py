"""Export concept OBJ meshes in metres; not watertight CFD/structural meshes."""
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'inputs/reference.json').read_text())['geometry']


def transform(p, beta, flip):
    x,y,z=p
    if flip:
        x,z=-x,-z
    a=math.radians(beta-90)
    return x, y*math.cos(a)-z*math.sin(a), y*math.sin(a)+z*math.cos(a)


def box(w,h,d,c=(0,0,0)):
    x,y,z=c
    vertices=[(x+sx*w/2,y+sy*h/2,z+sz*d/2)
              for sx,sy,sz in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                               (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]]
    faces=[(1,4,3,2),(5,6,7,8),(1,2,6,5),(4,8,7,3),(1,5,8,4),(2,3,7,6)]
    return vertices,faces


def export(mode,beta):
    groups=[]
    def add(name,mat,w,h,d,c=(0,0,0),move=False):
        v,f=box(w,h,d,c)
        if move:
            v=[transform(p,beta,mode=='dew') for p in v]
        groups.append((name,mat,v,f))
    w,h=CFG['width_m'],CFG['length_m']
    add('facade_section','wall',1.65,2.0,.08,(0,-.1,-.74))
    for x in [-.58,.58]:
        add('standoff','frame',.04,.04,.68,(x,0,-.35))
    add('thermal_core','insulation',w,h,.05,move=True)
    add('pv_face','pv',w,h,.005,(0,0,.0275),move=True)
    add('dew_face','dew',w,h,.005,(0,0,-.0275),move=True)
    for x in [-.53,.53]:
        add('side_frame','frame',.035,1.09,.09,(x,0,0),move=True)
    for y in [-.53,.53]:
        add('edge_frame','frame',1.09,.035,.09,(0,y,0),move=True)
    add('drain_channel_concept','water',1.15,.04,.10,(0,-.58,0),move=True)
    add('drain_riser_concept','water',.035,1.0,.035,(.72,-.72,-.10))
    add('tank_concept','frame',.38,.36,.28,(.62,-1.20,-.12))
    lines=['# Concept only; SI metres; no empirical validation','mtllib facade.mtl']
    offset=0
    for name,mat,v,f in groups:
        lines.extend(['o '+name,'usemtl '+mat])
        lines.extend('v '+' '.join(f'{q:.7f}' for q in p) for p in v)
        lines.extend('f '+' '.join(str(i+offset) for i in face) for face in f)
        offset+=len(v)
    dest=ROOT/'models'/f'facade-{mode}.obj'
    dest.write_text('\n'.join(lines)+'\n')
    return {'file':str(dest.relative_to(ROOT)),'vertices':offset,'tilt_deg':beta,
            'active_face_area_m2':w*h,'wall_standoff_m':CFG['wall_standoff_m']}


if __name__=='__main__':
    (ROOT/'models').mkdir(exist_ok=True)
    mats={'wall':(.6,.63,.66),'frame':(.45,.49,.55),'insulation':(.82,.76,.61),
          'pv':(.10,.23,.43),'dew':(.55,.8,.86),'water':(.2,.55,.8)}
    txt=[]
    for name,col in mats.items():
        txt.extend(['newmtl '+name,'Kd '+' '.join(map(str,col)),'illum 1',''])
    (ROOT/'models/facade.mtl').write_text('\n'.join(txt))
    manifest=[export(m,b) for m,b in CFG['illustrative_tilts_deg'].items()]
    (ROOT/'models/manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
