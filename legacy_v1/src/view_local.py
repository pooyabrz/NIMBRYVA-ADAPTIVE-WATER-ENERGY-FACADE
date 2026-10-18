"""LOCAL native Matplotlib 3D application. No web app or in-chat rendering.

Mouse drag rotates the scene. Controls share geometry/tilts with the simulator.
Rendered rays and plumbing are conceptual; they are not CFD or engineering CAD.
"""
import argparse
import math
import numpy as np
from geometry import panel_corners,panel_frame,view_factors
from weather import cities,load_weather,model_config


class LocalViewer:
    def __init__(self,city='rasht',year=2024):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider,RadioButtons,CheckButtons,Button
        self.plt=plt;self.cfg=model_config();self.year=year
        self.registry=cities();self.city_id=city
        self.cache={};self.frame=self._load(city)
        self.fig=plt.figure(figsize=(14,9))
        self.fig.canvas.manager.set_window_title('Adaptive Water Facade | LOCAL Python research viewer')
        self.ax=self.fig.add_axes([.25,.23,.72,.70],projection='3d')
        self.fig.text(.28,.96,'Adaptive water facade - local 3D research viewer',fontsize=16,weight='bold')
        self.fig.text(.28,.925,'Concept geometry | unvalidated heat model | rays are illustrative',fontsize=10,color='#a54720')
        self.info=self.fig.text(.27,.13,'',fontsize=9,family='monospace',va='top')
        self.result=self.fig.text(.27,.035,'Use mouse drag to rotate. Water is untreated collected water.',fontsize=9)
        self.city_radio=RadioButtons(self.fig.add_axes([.015,.40,.20,.51]),[c['name'] for c in self.registry],
                                    active=[c['id'] for c in self.registry].index(city))
        self.city_radio.on_clicked(self._city_changed)
        self.face_radio=RadioButtons(self.fig.add_axes([.025,.25,.18,.11]),['PV surface','Dew surface'])
        self.face_radio.on_clicked(lambda _:self.redraw())
        self.checks=CheckButtons(self.fig.add_axes([.02,.08,.20,.13]),['Separate layers','Sun vector','Rain vector'],[False,True,True])
        self.checks.on_clicked(lambda _:self.redraw())
        self.tilt=Slider(self.fig.add_axes([.34,.21,.54,.018]),'Tilt (deg)',20,80,valinit=50,valstep=[20,35,50,65,80])
        self.hour=Slider(self.fig.add_axes([.34,.18,.54,.018]),'UTC hour',0,len(self.frame)-1,valinit=4368,valstep=1,valfmt='%0.0f')
        self.height=Slider(self.fig.add_axes([.34,.15,.54,.018]),'Mount (m)',2,28,valinit=10,valstep=1)
        for slider in (self.tilt,self.hour,self.height):slider.on_changed(lambda _:self.redraw())
        self.button=Button(self.fig.add_axes([.025,.018,.19,.035]),'24 h reference calculation')
        self.button.on_clicked(self.estimate)
        self.ax.view_init(elev=22,azim=-58)
        self.redraw()

    def _load(self,city):
        if city not in self.cache:self.cache[city]=load_weather(city,self.year,self.cfg)
        return self.cache[city]

    def _city_changed(self,name):
        self.city_id=next(c['id'] for c in self.registry if c['name']==name)
        self.frame=self._load(self.city_id)
        self.result.set_text('City changed. Any calculation uses a fresh 24 h warm-up.')
        self.redraw()

    def redraw(self):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        ax=self.ax;elev,azim=ax.elev,ax.azim;ax.clear()
        beta=float(self.tilt.val);height=float(self.height.val);hour=int(self.hour.val)
        row=self.frame.iloc[hour];u,v,n=panel_frame(beta)
        active=0 if self.face_radio.value_selected=='PV surface' else 1
        exploded,sun_on,rain_on=self.checks.get_status()
        layer_gap=.14 if exploded else .025
        colors=['#164c82','#dde6ed']
        for face,offset in [(1-active,-layer_gap),(active,layer_gap)]:
            quad=panel_corners(beta,self.cfg['area_m2'],height,offset)
            ax.add_collection3d(Poly3DCollection([quad],facecolors=colors[face],edgecolors='#516478',linewidths=1.2,alpha=.94))
            if face==0:
                centre=np.array([0.,height,0.])+offset*n
                for k in np.linspace(-.5,.5,7):
                    line=np.array([centre+k*u-.5*v,centre+k*u+.5*v])
                    ax.plot(*line.T,color='#8cc7e8',lw=.5)
                for k in np.linspace(-.5,.5,5):
                    line=np.array([centre-.5*u+k*v,centre+.5*u+k*v])
                    ax.plot(*line.T,color='#8cc7e8',lw=.5)
        # A cropped facade patch; full building dimensions still set view factors.
        wall=np.array([[-1.2,height-1.2,-.7],[1.2,height-1.2,-.7],[1.2,height+1.2,-.7],[-1.2,height+1.2,-.7]])
        ax.add_collection3d(Poly3DCollection([wall],facecolors='#acb4bc',alpha=.30,edgecolors='#788793'))
        centre=np.array([0.,height,0.])
        for x in (-.42,.42):
            ax.plot([x,x],[height,height],[-.7,0],color='#697683',lw=3)
        # Bottom edge / symbolic gutter and flexible drainage route.
        low=centre-.5*v
        gutter=np.array([low-.55*u,low+.55*u])
        ax.plot(*gutter.T,color='#128e97',lw=5)
        pipe=np.array([low+.55*u,[.70,height-.65,-.35],[.70,height-1.1,-.50]])
        ax.plot(*pipe.T,color='#128e97',lw=3)
        ax.quiver(*centre,*n,length=.85,color='#259566',arrow_length_ratio=.14)
        ax.text(*(centre+.88*n),'Active normal',fontsize=8,color='#258050')
        if sun_on and row.sun_y>0:
            sun=np.array([row.sun_x,row.sun_y,row.sun_z])
            start=centre+sun*1.15
            ax.quiver(*start,*(-sun),length=.8,color='#edb42b',arrow_length_ratio=.12)
            ax.text(*start,'Sun',color='#b88010',fontsize=9)
        if rain_on and row.rain_mm_h>0:
            rain=np.array([row.wind_x,-self.cfg['drop_fall_speed_m_s'],row.wind_z]);rain/=np.linalg.norm(rain)
            for x in (-.4,0,.4):
                start=centre+np.array([x,1.,.6])
                ax.quiver(*start,*rain,length=.60,color='#2d9cda',arrow_length_ratio=.13)
        fs,fg,fw=view_factors(beta,height,.7,self.cfg['building_height_m'],self.cfg['building_width_m'])
        ax.set(xlim=(-1.2,1.2),ylim=(height-1.2,height+1.3),zlim=(-.9,1.25),
               xlabel='x: facade lateral (m)',ylabel='y: above ground (m)',zlabel='z: outward (m)')
        ax.set_box_aspect((2.4,2.5,2.15));ax.view_init(elev=elev,azim=azim)
        ax.set_title(f"{self.frame.attrs['city']['name']} | {self.frame.index[hour]:%Y-%m-%d %H:%M} UTC",fontsize=11)
        self.info.set_text(f"Air {row.air_C:.1f} C | RH {row.rh_pct:.0f}% | rain {row.rain_mm_h:.2f} mm/h | wind {row.wind_m_s:.1f} m/s\n"
                           f"Site elevation {self.frame.attrs['site_elevation_m']:.0f} m != mount height {height:.0f} m | pressure {row.pressure_Pa/100:.0f} hPa\n"
                           f"Sky / ground / wall view: {fs:.3f} / {fg:.3f} / {fw:.3f} | LW down {row.lw_down_W_m2:.0f} W/m2")
        self.fig.canvas.draw_idle()

    def estimate(self,_=None):
        from physics import PanelModel
        cfg=dict(self.cfg);cfg['mounting_height_m']=float(self.height.val)
        frame=load_weather(self.city_id,self.year,cfg)
        end=max(1,int(self.hour.val));start=max(0,end-24)
        model=PanelModel(cfg);model.reset(frame.iloc[start].air_C)
        tilt=cfg['tilts_deg'].index(int(self.tilt.val))
        face=0 if self.face_radio.value_selected=='PV surface' else 1
        invalid=False
        for i in range(start+1,end+1):
            result=model.step_hour(frame.iloc[i],face*len(cfg['tilts_deg'])+tilt)
            invalid |= not bool(result['water_supported'])
        if invalid:
            message='UNSUPPORTED cold/snow state in warm-up: no defensible liquid-water estimate.'
        else:
            message=(f"UNVALIDATED last-hour reference after {end-start} h: drained {result['drain_L']:.4f} L | "
                     f"PV {result['pv_gross_Wh']:.1f} Wh | active surface {model.state.temperature_C[face]:.1f} C")
        self.result.set_text(message);self.fig.canvas.draw_idle()


def main():
    p=argparse.ArgumentParser();p.add_argument('--city',default='rasht');p.add_argument('--year',type=int,default=2024)
    p.add_argument('--check',action='store_true',help='Exercise geometry and callbacks headlessly; do not display a scene')
    a=p.parse_args()
    if a.check:
        import matplotlib
        matplotlib.use('Agg')
    app=LocalViewer(a.city,a.year)
    if a.check:
        app.tilt.set_val(20);app.height.set_val(15);app.hour.set_val(4800)
        app.face_radio.set_active(1);app.checks.set_active(0)
        app.city_radio.set_active(4);app.estimate()
        app.fig.canvas.draw()
        assert 'UNVALIDATED' in app.result.get_text() or 'UNSUPPORTED' in app.result.get_text()
        app.plt.close(app.fig)
        print('Local 3D viewer: callbacks and geometry checked; no image displayed.')
    else:
        app.plt.show()


if __name__=='__main__':main()
