"""Desktop-only Python 3D cassette viewer. No web or in-chat scene."""
import argparse,json
import numpy as np
from weather import ROOT,cities,load_weather,model_config
from geometry import panel_frame,rectangle_corners,tiled_corners
from assembly import compile_assembly
from actuator import pose,transition,motion_angle


class LocalViewer:
    def __init__(self,city='rasht',year=2024):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider,RadioButtons,CheckButtons,Button
        self.plt=plt;self.cfg=model_config();self.assembly=compile_assembly(self.cfg);self.year=year
        self.city_id=city;self.cache={};self.frame=self._load(city);self.motion_override=None
        self.fig=plt.figure(figsize=(14,9));self.fig.canvas.manager.set_window_title('Cassette v2 | LOCAL Python')
        self.ax=self.fig.add_axes([.25,.27,.72,.66],projection='3d')
        self.fig.text(.27,.96,'Adaptive water facade | layered cassette v2',fontsize=15,weight='bold')
        self.fig.text(.27,.925,'Proposed geometry. Air arrows and collector are concepts; no CFD/structural qualification.',fontsize=9,color='#a54720')
        self.info=self.fig.text(.27,.12,'',fontsize=9,va='top')
        self.result=self.fig.text(.27,.025,'Drag to rotate. Native Python window; no cloud visualization.',fontsize=9)
        registry=cities();self.registry=registry
        self.city_radio=RadioButtons(self.fig.add_axes([.01,.43,.21,.47]),[c['name'] for c in registry],active=[c['id'] for c in registry].index(city))
        self.city_radio.on_clicked(self._city_changed)
        self.face_radio=RadioButtons(self.fig.add_axes([.02,.30,.20,.10]),['PV upward','Emitter upward'])
        self.face_radio.on_clicked(lambda _:self.redraw())
        self.checks=CheckButtons(self.fig.add_axes([.02,.20,.20,.08]),['Explode layers','Air/condensate path'],[False,True])
        self.checks.on_clicked(lambda _:self.redraw())
        self.tilt=Slider(self.fig.add_axes([.34,.245,.54,.018]),'Tilt (deg)',20,80,valinit=50,valstep=[20,35,50,65,80])
        self.hour=Slider(self.fig.add_axes([.34,.215,.54,.018]),'UTC hour',0,len(self.frame)-1,valinit=4800,valstep=1,valfmt='%0.0f')
        self.height=Slider(self.fig.add_axes([.34,.185,.54,.018]),'Mount (m)',2,10,valinit=8,valstep=.5)
        for x in [self.tilt,self.hour,self.height]:x.on_changed(lambda _:self.redraw())
        self.button=Button(self.fig.add_axes([.02,.11,.20,.045]),'24 h conditional calculation');self.button.on_clicked(self.estimate)
        self.flip_button=Button(self.fig.add_axes([.02,.045,.20,.045]),'Preview 180 deg motion');self.flip_button.on_clicked(self.animate)
        self.ax.view_init(elev=22,azim=-58);self.redraw()

    def _load(self,city):
        if city not in self.cache:self.cache[city]=load_weather(city,self.year,self.cfg)
        return self.cache[city]

    def _city_changed(self,name):
        self.city_id=next(c['id'] for c in self.registry if c['name']==name);self.frame=self._load(self.city_id)
        self.result.set_text('City changed; calculate again for the selected boundary conditions.');self.redraw()

    def redraw(self):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        ax=self.ax;elev,azim=ax.elev,ax.azim;ax.clear();c=self.cfg
        beta=float(self.tilt.val);height=float(self.height.val);face=int(self.face_radio.value_selected=='Emitter upward')
        angle=pose(face,beta) if self.motion_override is None else self.motion_override
        u,v,n=panel_frame(angle);row=self.frame.iloc[int(self.hour.val)]
        exploded,arrows=self.checks.get_status();gap=.10 if exploded else 0
        # Correct physical side ordering: PV front -> cell -> PV back -> XPS
        # -> open humid-air gap -> underside CNF -> silica -> outer PDMS.
        layers=[('PV front glass','#426b8b',.035),('PV cell','#103f66',.032),('PV back glass','#9babb6',.029),
                ('XPS inner wall','#dfd0a5',.005),('CNF underside','#414b4c',-.034),
                ('Silica + outer PDMS','#cee8ec',-.035)]
        for j,(name,color,offset) in enumerate(layers):
            offset+=(2.5-j)*gap
            w=c['panel_width_m'];ch=c['rotation_chord_m']
            if j>=4:w=1.09;ch=1.75
            quads=tiled_corners(angle,height,offset) if j>=4 else [rectangle_corners(angle,w,ch,height,offset)]
            ax.add_collection3d(Poly3DCollection(quads,facecolors=color,edgecolors='#56616a',alpha=.75 if j==0 else .92,linewidths=.4))
            if exploded:ax.text(*(np.array([w/2+.02,height,0])+offset*n),name,fontsize=7)
            if j==1:
                centre=np.array([0,height,0])+offset*n
                for x in np.linspace(-w/2,w/2,7):
                    line=np.array([centre+x*u-ch/2*v,centre+x*u+ch/2*v]);ax.plot(*line.T,color='#8cc7e8',lw=.5)
        standoff=c['wall_standoff_m']
        wall=np.array([[-1.25,height-1.2,-standoff],[1.25,height-1.2,-standoff],[1.25,height+1.2,-standoff],[-1.25,height+1.2,-standoff]])
        ax.add_collection3d(Poly3DCollection([wall],facecolors='#b5bdc5',alpha=.25))
        for x in [-.45,.45]:ax.plot([x,x],[height,height],[-standoff,0],color='#64727d',lw=4)
        centre=np.array([0.,height,0.]);ax.quiver(*centre,*n,length=.7,color='#288b60');ax.text(*(centre+.72*n),'PV normal',fontsize=8)
        if arrows:
            for x in [-.3,0,.3]:
                start=centre+x*u-.75*v-.02*n;ax.quiver(*start,*v,length=1.4,color='#25a2a2',arrow_length_ratio=.12)
            low=centre-.88*v;line=np.array([low-.55*u,low+.55*u]);ax.plot(*line.T,color='#177f9b',lw=4)
            ax.text(*low,'Proposed drain; capture not calibrated',fontsize=7)
        clearance=c['wall_standoff_m']-np.hypot(c['rotation_chord_m']/2,self.assembly['assembly_depth_m']/2)
        ax.set(xlim=(-1.25,1.25),ylim=(height-1.25,height+1.25),zlim=(-1.1,1.2),xlabel='lateral x (m)',ylabel='height y (m)',zlabel='outward z (m)')
        ax.set_box_aspect((2.5,2.5,2.3));ax.view_init(elev=elev,azim=azim)
        ax.set_title(f"{self.frame.attrs['city']['name']} | {self.frame.index[int(self.hour.val)]:%Y-%m-%d %H:%M} UTC",fontsize=11)
        self.info.set_text(f"PV {c['panel_width_m']:.3f} x {c['rotation_chord_m']:.3f} m | dew active {c['dew_area_m2']:.2f} m2 | min wall sweep gap {clearance:.3f} m\n"
          f"Angle {angle:.1f} deg | mount {height:.1f} m | geographical elevation {self.frame.attrs['site_elevation_m']:.0f} m\n"
          f"Reference air {row.air_C:.1f} C | RH {row.rh_pct:.1f}% | wind {row.wind_m_s:.1f} m/s | LW {row.lw_down_W_m2:.1f} W/m2\n"
          'Small layer separations are exaggerated. A complete motion can wet or drain different surfaces.')
        self.fig.canvas.draw_idle()

    def estimate(self,_=None):
        from physics import PanelModel
        cfg=dict(self.cfg);cfg['mounting_height_m']=float(self.height.val)
        frame=load_weather(self.city_id,self.year,cfg);end=max(1,int(self.hour.val));start=max(0,end-24)
        model=PanelModel(cfg);model.reset(frame.iloc[start].air_C);face=int(self.face_radio.value_selected=='Emitter upward')
        action=face*len(cfg['tilts_deg'])+cfg['tilts_deg'].index(int(self.tilt.val));bad=False
        for i in range(start+1,end+1):
            r=model.step_hour(frame.iloc[i],action);bad|=not bool(r['water_supported'])
        self.result.set_text('UNSUPPORTED liquid-water regime in this window.' if bad else
           f"CONDITIONAL last hour: drain {r['drain_L']:.4f} L | PV {r['pv_gross_Wh']:.1f} Wh | cell {r['cell_C']:.1f} C | dew {r['surface_dew_C']:.1f} C. NOT validated yield.")
        self.fig.canvas.draw_idle()

    def animate(self,_=None):
        from matplotlib.animation import FuncAnimation
        beta=float(self.tilt.val);face=int(self.face_radio.value_selected=='Emitter upward')
        move=transition(self.cfg,pose(face,beta),pose(1-face,beta))
        def frame(i):
            self.motion_override=motion_angle(move,move['duration_s']*i/59)
            self.redraw();self.result.set_text(f"Accelerated preview | physical time {move['duration_s']*i/59:.1f}/{move['duration_s']:.1f} s; wind veto not evaluated in preview.")
            if i==59:self.motion_override=None;self.face_radio.set_active(1-face)
            return []
        self.animation=FuncAnimation(self.fig,frame,frames=60,interval=60,repeat=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--city',default='rasht');p.add_argument('--year',type=int,default=2024);p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:
        import matplotlib
        matplotlib.use('Agg')
    app=LocalViewer(a.city,a.year)
    if a.check:
        app.tilt.set_val(20);app.height.set_val(9);app.face_radio.set_active(1);app.checks.set_active(0)
        app.city_radio.set_active(4);app.estimate();app.fig.canvas.draw();app.plt.close(app.fig)
        out=ROOT/'results/cassette-v2';out.mkdir(parents=True,exist_ok=True)
        (out/'viewer-check.json').write_text(json.dumps({'status':'passed','headless_callbacks_exercised':True,'rendered_image_displayed_or_delivered':False}))
        print('Local Python viewer checked headlessly; no scene/image displayed.')
    else:app.plt.show()


if __name__=='__main__':main()
