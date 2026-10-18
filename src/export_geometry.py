"""Optional local OBJ export using the same v2 rectangle/rotation convention.

The mesh is only a layered concept, not a watertight CFD/structural domain.
"""
import json
from geometry import rectangle_corners,tiled_corners
from assembly import ROOT,compile_assembly
from weather import model_config


def export(face,beta):
    c=model_config();a=compile_assembly(c);angle=beta+180*face;depth=a['assembly_depth_m']
    layers=[('pv_front',depth/2,'pv'),('pv_back',depth/2-.004,'frame'),('xps_inner',-.014,'insulation'),('cnf_inner',-depth/2+.0006,'dew'),('emitter_outer',-depth/2,'dew')]
    text=['# cassette-v2.0; SI metres; conceptual sheets, not CFD mesh','mtllib facade.mtl'];count=0
    for name,offset,material in layers:
        quads=tiled_corners(angle,c['mounting_height_m'],offset) if name in ['cnf_inner','emitter_outer'] else [rectangle_corners(angle,c['panel_width_m'],c['rotation_chord_m'],c['mounting_height_m'],offset)]
        for i,corners in enumerate(quads):
            text.extend(['o '+name+str(i),'usemtl '+material]);text.extend('v '+' '.join(f'{v:.8f}' for v in x) for x in corners)
            text.append('f '+' '.join(str(count+i) for i in [1,2,3,4]));count+=4
    out=ROOT/'models';out.mkdir(exist_ok=True);dest=out/f'cassette-v2-face{face}-beta{beta}.obj';dest.write_text('\n'.join(text)+'\n')
    return {'path':str(dest.relative_to(ROOT)),'model_version':c['model_version'],'face':face,'beta_deg':beta,'vertices':count,'not_a_solver_mesh':True}


if __name__=='__main__':
    out=ROOT/'models';out.mkdir(exist_ok=True)
    materials={'pv':(.1,.25,.42),'frame':(.4,.45,.5),'insulation':(.8,.75,.6),'dew':(.7,.88,.9)}
    (out/'facade.mtl').write_text('\n'.join('newmtl '+name+'\nKd '+' '.join(map(str,color))+'\nillum 1\n' for name,color in materials.items()))
    (out/'manifest.json').write_text(json.dumps([export(f,50) for f in [0,1]],indent=2))
    print('Local concept OBJ files created. No image displayed.')
