"""Native Python 3-D replay + recorded numerical outputs; no cloud rendering."""
import argparse,json
import numpy as np
import pandas as pd
from assembly import ROOT
from weather import model_config
from geometry import panel_frame

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',choices=['emulator','teacher','presentation'],default='teacher');p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:
        import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider,RadioButtons
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from stageb.rain import triangles
    file=ROOT/'results/research/E/control'/('teacher-replay-hourly.csv.gz' if a.source=='teacher' else 'test-hourly.csv.gz')
    if a.source=='presentation':file=ROOT/'results/presentation/hourly-audited.csv.gz'
    f=pd.read_csv(file);c=model_config();groups=sorted(f.group.unique());kinds=[k for k in ['fixed','emitter','rule','mpc','ppo'] if k in f.controller.unique()];selected={'group':groups[0],'controller':'fixed'}
    fig=plt.figure(figsize=(14,8));ax=fig.add_subplot(221,projection='3d');water=fig.add_subplot(222);energy=fig.add_subplot(224);note=fig.add_subplot(223);note.axis('off')
    fig.subplots_adjust(left=.20,bottom=.14,hspace=.35);mesh=json.loads((ROOT/'results/stage-b/cfd/reference-medium/case.json').read_text());building=np.array(mesh['building_vertices_m'])
    slider=Slider(fig.add_axes([.27,.04,.65,.035]),'Hour',0,47,valinit=0,valstep=1)
    panel=RadioButtons(fig.add_axes([.01,.72,.16,.18]),kinds);locations=RadioButtons(fig.add_axes([.005,.13,.18,.55]),groups)
    def draw(_=None):
        z=f[(f.group==selected['group'])&(f.controller==selected['controller'])].reset_index(drop=True);i=min(int(slider.val),len(z)-1);r=z.iloc[i]
        action=int(r.get('resolved_action',r.get('requested_action',2)));face,tilt=divmod(action,5)
        angle=float(r.get('angle_deg',c['tilts_deg'][tilt]+180*face));u,v,n=panel_frame(angle);points=[]
        for sx,sy,sz in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]:
            pos=np.array([0.,c['mounting_height_m'],0.])+sx*c['panel_width_m']/2*u+sy*c['rotation_chord_m']/2*v+sz*.070602/2*n
            points.append([-pos[2]-c['wall_standoff_m'],pos[0],pos[1]])
        ax.clear();ax.add_collection3d(Poly3DCollection(triangles(building),facecolors='grey',edgecolors='grey',alpha=.12))
        ax.add_collection3d(Poly3DCollection(triangles(np.array(points)),facecolors='royalblue' if face==0 else 'teal',edgecolors='black',alpha=.85))
        ax.set(xlim=(-3,2),ylim=(-3,3),zlim=(5,11),xlabel='X [m]',ylabel='Y [m]',zlabel='Height [m]',title=f'{selected["controller"]} | angle {angle:.0f} deg')
        ax.set_box_aspect([5,6,6]);water.clear();energy.clear()
        for kind in kinds:
            q=f[(f.group==selected['group'])&(f.controller==kind)].reset_index(drop=True)
            if 'common_valid_prefix' in q:
                xx=np.arange(len(q))+1;ok=q.common_valid_prefix.astype(bool).to_numpy();water.plot(xx,q.drain_L.cumsum(),ls=':',alpha=.25);energy.plot(xx,q.pv_gross_Wh.cumsum(),ls=':',alpha=.25)
                water.plot(xx[ok],q.drain_L.cumsum().to_numpy()[ok],label=kind);energy.plot(xx[ok],q.pv_gross_Wh.cumsum().to_numpy()[ok],label=kind)
            else:water.plot(np.arange(len(q))+1,q.drain_L.cumsum(),label=kind);energy.plot(np.arange(len(q))+1,q.pv_gross_Wh.cumsum(),label=kind)
        for chart,label in [(water,'Cumulative drain [L]'),(energy,'Cumulative gross PV [Wh]')]:
            chart.axvline(i+1,color='grey',ls=':');chart.set(xlabel='Realized hour',ylabel=label);chart.grid(alpha=.2);chart.legend(fontsize=8)
        prefix_note=f'\nCommon scored prefix: {bool(r.common_valid_prefix)}\nDotted curves: diagnostic only' if 'common_valid_prefix' in r else ''
        note.clear();note.axis('off');note.text(0,.95,f'{selected["group"]}\n{r.time}\nPV surface: {r.surface_pv_C:.2f} C\nEmitter: {r.surface_dew_C:.2f} C\nBattery: {r.battery_Wh:.2f} Wh\nLiquid regime flag: {int(r.water_supported)}{prefix_note}\n\nConditional numerical model replay\nNo experimental product qualification\nNo future reanalysis given to controllers',va='top',fontsize=10)
        fig.suptitle('C/D/E local Python replay | '+a.source+' | unqualified yield',fontsize=12);fig.canvas.draw_idle()
    def controller(label):selected['controller']=label;draw()
    def location(label):selected['group']=label;draw()
    panel.on_clicked(controller);locations.on_clicked(location);slider.on_changed(draw);draw()
    if a.check:
        for kind in kinds:selected['controller']=kind;draw();fig.canvas.draw()
        slider.set_val(30);fig.canvas.draw();plt.close(fig)
        dest=ROOT/('results/presentation/view-check.json' if a.source=='presentation' else 'results/research/local-view-check.json');dest.write_text(json.dumps({'pass':True,'source':a.source,'native_python':True,'3D_image_file_created':False}));print(dest.read_text())
    else:plt.show()
if __name__=='__main__':main()
