"""Local 360-degree geometry and drainage-state demo.

This module does NOT run the thermal/PV/CFD plant or estimate water yield.
Water inputs, animation speed, transfer rate and dock timing are scripted.
"""
from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DemoController:
    theta: float = 0.0  # unwrapped degrees, 0 = PV outward
    volume: float = 0.0
    storage: float = 0.0
    input_total: float = 0.0
    bypass: float = 0.0
    capacity: float = 5.0
    storage_capacity: float = 20.0  # display assumption, not a product specification
    residual: float = 0.2
    state: str = "IDLE"
    locked: bool = True
    dock_engaged: bool = False
    inlet_open: bool = False
    vent_open: bool = False
    drain_open: bool = False
    wind_locked: bool = False  # injected test flag, not a wind-speed model
    target: float | None = None
    pending_target: float | None = None
    timer: float = 0.0
    moves: int = 0
    discharge_cycles: int = 0
    event: str = "Ready. Water inputs and rates are illustrative."

    def at_dock(self):
        return abs((self.theta - 180 + 180) % 360 - 180) < 1e-6

    def close_ports(self):
        self.inlet_open = self.vent_open = self.drain_open = False

    def park(self):
        self.state = "IDLE"
        self.locked = True
        self.close_ports()
        if self.at_dock() and not self.dock_engaged:
            self.inlet_open = self.vent_open = True

    def request_move(self, target):
        if not math.isfinite(target):
            self.event = "Blocked: target angle must be finite."
            return False
        if self.state != "IDLE":
            self.event = "Blocked: finish the current movement or discharge."
            return False
        if self.wind_locked:
            self.event = "BLOCKED: injected wind interlock. No wind rating is implied."
            return False
        if self.dock_engaged:
            self.event = "Blocked: the coupling is still engaged."
            return False
        if abs(target - self.theta) < 1e-9:
            self.event = "Already at requested pose."
            return True
        if self.volume > self.residual + 1e-9:
            if not self.at_dock():
                self.event = "Blocked: a water-loaded return path is not validated."
                return False
            self.pending_target = float(target)
            if not self.request_drain("Drain before repositioning"):
                self.pending_target = None
                return False
            return True
        self.close_ports()
        self.target = float(target)
        self.locked = False
        self.state = "MOVING"
        self.moves += 1
        self.event = "Moving with closed ports and a retracted dock (logic demo)."
        return True

    def request_face(self, collector):
        pose = 180.0 if collector else 0.0
        delta = (pose - self.theta + 180) % 360 - 180
        return self.request_move(self.theta + delta)

    def request_drain(self, reason="Manual discharge request"):
        if self.state != "IDLE" or not self.at_dock() or not self.locked:
            self.event = "Blocked: discharge needs the parked 180-degree dock pose."
            return False
        need = max(0.0, self.volume - self.residual)
        if need <= 1e-9:
            self.event = "Buffer already at or below the residual limit."
            return False
        if self.storage_capacity - self.storage + 1e-9 < need:
            self.close_ports()
            self.event = "Blocked: insufficient receiving capacity. Intake isolated."
            return False
        self.close_ports()
        self.timer = 0.0
        self.state = "CONNECTING"
        self.event = reason + ": locking and connecting."
        return True

    def add_water(self, liters=1.0):
        if not math.isfinite(liters) or liters < 0:
            raise ValueError("Scripted water must be finite and nonnegative")
        self.input_total += liters
        if self.state != "IDLE" or not self.at_dock() or not self.inlet_open:
            self.bypass += liters
            self.event = "Scripted input bypassed: collection pose/intake unavailable."
            self.check_invariants()
            return
        accepted = min(liters, self.capacity - self.volume)
        self.volume += accepted
        self.bypass += liters - accepted
        self.event = "Scripted water added. This is not a weather-derived yield."
        if self.volume >= 4.5:
            self.request_drain("4.5 L high-high request")
        elif self.volume >= 4.0:
            self.request_drain("4.0 L automatic discharge request")
        self.check_invariants()

    def toggle_wind(self):
        self.wind_locked = not self.wind_locked
        if self.wind_locked and self.state == "MOVING":
            self.target = self.pending_target = None
            self.close_ports()
            self.state = "IDLE"
            self.locked = True
            self.event = "Wind interlock stops demo motion. Braking physics is absent."
        else:
            self.event = "Injected wind interlock " + ("ON" if self.wind_locked else "OFF")

    def advance(self, dt):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("Time step must be finite and nonnegative")
        if self.state == "MOVING":
            step = min(abs(self.target - self.theta), 45.0 * dt)
            self.theta += math.copysign(step, self.target - self.theta)
            if abs(self.target - self.theta) < 1e-8:
                self.theta = self.target
                self.target = None
                self.park()
                self.event = "Parked. Dock pose." if self.at_dock() else "Parked. PV / transition pose."
                if self.at_dock() and self.volume >= 1.0:
                    self.request_drain("Dock-arrival discharge event")
        elif self.state == "CONNECTING":
            self.timer += dt
            if self.timer >= 0.7:
                self.dock_engaged = True
                self.vent_open = self.drain_open = True
                self.state = "DRAINING"
                self.event = "Transfer to fixed storage. Demonstration rate: 1 L/s."
        elif self.state == "DRAINING":
            transfer = min(dt, max(0.0, self.volume - self.residual), self.storage_capacity - self.storage)
            self.volume -= transfer
            self.storage += transfer
            if self.volume <= self.residual + 1e-9:
                self.volume = max(0.0, self.volume)
                self.close_ports()
                self.state = "DISCONNECTING"
                self.timer = 0.0
                self.event = "Liquid and vent valves closed. Retracting coupling."
                self.discharge_cycles += 1
        elif self.state == "DISCONNECTING":
            self.timer += dt
            if self.timer >= 0.7:
                self.dock_engaged = False
                target = self.pending_target
                self.pending_target = None
                self.park()
                self.event = "Discharge finished. Dock retracted."
                if target is not None:
                    self.request_move(target)
        self.check_invariants()

    def check_invariants(self):
        assert -1e-9 <= self.volume <= self.capacity + 1e-9
        assert -1e-9 <= self.storage <= self.storage_capacity + 1e-9
        assert abs(self.input_total - self.volume - self.storage - self.bypass) < 1e-8
        if self.state == "MOVING":
            assert not self.locked and not self.dock_engaged and not self.wind_locked
            assert not (self.inlet_open or self.vent_open or self.drain_open)
            assert self.volume <= self.residual + 1e-9
        if self.dock_engaged:
            assert self.locked and self.at_dock() and self.state != "MOVING"
        if self.state == "DRAINING":
            assert self.dock_engaged and self.vent_open and self.drain_open


def settle(model, seconds=30):
    for _ in range(int(seconds / 0.05)):
        model.advance(0.05)
        if model.state == "IDLE":
            return
    raise RuntimeError("Demo state did not settle")


def geometry_check():
    radius = math.hypot(1.115, 0.04)
    return {"swept_radius_m": radius, "minimum_wall_gap_m": 1.40 - radius,
            "fixed_dock_gap_m": 1.20 - radius,
            "physical_validation": False, "scope": "Rigid shell with flush ports only"}


def create_app():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    model = DemoController()
    fig = plt.figure(figsize=(13.4, 7.4), facecolor="white")
    ax = fig.add_axes([.025, .22, .67, .70], projection="3d")
    info = fig.add_axes([.73, .22, .255, .68]); info.axis("off")
    ax.view_init(elev=16, azim=-48)
    fig.suptitle("Nimbryva | 360-degree geometry + drainage-control demo", fontsize=17, color="#083e49")
    fig.text(.04, .17, "Scripted water and timing. No thermal, PV-yield, CFD or structural prediction.", fontsize=11, color="#526c75")
    buttons = {}

    def box(bounds, color, rotate=False):
        x0,x1,y0,y1,z0,z1=bounds
        v=np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                    [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
        if rotate:
            a=math.radians(model.theta);u=v.copy()
            v[:,0]=1.4+u[:,0]*math.cos(a)-u[:,2]*math.sin(a)
            v[:,2]=u[:,0]*math.sin(a)+u[:,2]*math.cos(a)
        faces=[[v[j] for j in ids] for ids in [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]]
        ax.add_collection3d(Poly3DCollection(faces,facecolors=color,edgecolors="#38515a",linewidths=.45))

    def draw():
        elevation,azimuth=ax.elev,ax.azim
        ax.clear()
        wall=[(0,-.85,-1.5),(0,.85,-1.5),(0,.85,1.5),(0,-.85,1.5)]
        ax.add_collection3d(Poly3DCollection([wall],facecolors="#dce8eb",edgecolors="#a9bec2",alpha=.22))
        for yy in [-.65,.65]:
            box((1.365,1.435,yy-.035,yy+.035,-1.3,1.3),"#083e49")
            for zz in [-1.3,1.3]:box((0,1.4,yy-.035,yy+.035,zz-.035,zz+.035),"#083e49")
            box((1.34,1.46,yy-.10,yy+.10,-.07,.07),"#f3b84f")
        for zz in [-1.3,1.3]:box((1.365,1.435,-.65,.65,zz-.035,zz+.035),"#083e49")
        box((-.03,.03,-.50,.50,-1,1),"#bfcdd0",True)
        box((.0301,.033,-.475,.475,-.98,.98),"#194b70",True)
        box((-.033,-.0301,-.475,.475,-.98,.98),"#f3efdf",True)
        box((-.04,.04,-.46,.46,1.015,1.115),"#0b989b",True)
        box((1.36,1.44,-.06,.06,-1.28,-1.20),"#7a969c")
        if model.dock_engaged:box((1.375,1.425,-.045,.045,-1.20,-1.115),"#0b989b")
        ax.plot([1.4,1.4,2.6],[0,0,1.1],[-1.28,-1.5,-1.65],color="#7a969c",lw=3)
        box((2.38,2.85,.90,1.35,-2.15,-1.67),"#b7c8cc")
        a=np.linspace(0,2*np.pi,160);r=geometry_check()["swept_radius_m"]
        ax.plot(1.4+r*np.cos(a),np.full_like(a,-.72),r*np.sin(a),color="#e4ac40",lw=1,ls="--")
        ax.set(xlim=(-.1,2.95),ylim=(-.95,1.5),zlim=(-2.2,1.5),xlabel="Outward from wall [m]",ylabel="Width [m]",zlabel="Height relative to pivot [m]")
        ax.set_box_aspect((3.05,2.45,3.7));ax.view_init(elev=elevation,azim=azimuth)
        info.clear();info.axis("off")
        entries=[("STATE",model.state),("Angle",f"{model.theta % 360:.1f} deg (periodic)"),
                 ("Moving buffer",f"{model.volume:.2f} / 5.00 L"),("Fixed storage",f"{model.storage:.2f} L"),
                 ("Coupling","ENGAGED" if model.dock_engaged else "retracted"),
                 ("Intake / vent",f"{'open' if model.inlet_open else 'closed'} / {'open' if model.vent_open else 'closed'}"),
                 ("Drain valve","open" if model.drain_open else "closed"),
                 ("Wind test flag","LOCKED" if model.wind_locked else "clear"),
                 ("Movements",str(model.moves)),("Discharge cycles",str(model.discharge_cycles)),
                 ("Scripted bypass",f"{model.bypass:.2f} L")]
        for i,(label,value) in enumerate(entries):
            yy=.98-i*.064
            info.text(0,yy,label,fontsize=10,color="#526c75",va="top")
            info.text(.52,yy,value,fontsize=10,color="#083e49",va="top",weight="bold")
        import textwrap
        info.text(0,.22,"\n".join(textwrap.wrap(model.event,43)),fontsize=11,color="#a63c30" if "BLOCKED" in model.event or "Blocked" in model.event else "#087b81",va="top")
        info.text(0,.02,"Capacity, thresholds, rates and\nreceiver size are demo assumptions.",fontsize=9,color="#526c75",va="bottom")
        fig.canvas.draw_idle()

    def reset():
        nonlocal model
        model=DemoController();draw()

    actions=[("Full turn",lambda:model.request_move(model.theta+360)),
             ("Collector",lambda:model.request_face(True)),("Solar",lambda:model.request_face(False)),
             ("Add 1 L",lambda:model.add_water(1)),("Drain",lambda:model.request_drain()),
             ("Wind lock",lambda:model.toggle_wind()),("Reset",reset)]
    for i,(label,callback) in enumerate(actions):
        b=Button(fig.add_axes([.035+i*.137,.075,.127,.055]),label,color="#eff7f7",hovercolor="#dceced")
        def click(_,callback=callback):callback();draw()
        b.on_clicked(click);buttons[label]=b
    last=time.monotonic()
    def tick():
        nonlocal last
        now=time.monotonic();dt=min(now-last,.25);last=now
        old=(model.state,model.theta,model.volume,model.storage)
        model.advance(dt)
        if old!=(model.state,model.theta,model.volume,model.storage):draw()
    timer=fig.canvas.new_timer(interval=60);timer.add_callback(tick);timer.start()
    fig._nimbryva_handles=(buttons,timer)
    draw()
    return fig,lambda:model,draw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check",action="store_true",help="Run geometry and state checks without a window")
    p.add_argument("--check-gui",action="store_true",help="Draw the local GUI headlessly without writing images")
    args=p.parse_args()
    if args.check:
        m=DemoController();assert m.request_move(360);settle(m)
        assert m.request_face(True);settle(m)
        for _ in range(4):m.add_water()
        settle(m);assert abs(m.volume-.2)<1e-8 and abs(m.storage-3.8)<1e-8
        m.toggle_wind();assert not m.request_move(m.theta+360)
        print(json.dumps({"checks":"passed","geometry":geometry_check(),"cycle_transferred_L":m.storage,"performance_model":False},indent=2))
        return
    if args.check_gui:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,get_model,draw=create_app()
    if args.check_gui:
        get_model().request_face(True);settle(get_model());draw();fig.canvas.draw();plt.close(fig)
        print(json.dumps({"GUI_check":"passed","3D_image_file_created":False,"performance_model":False}))
    else:plt.show()


if __name__=="__main__":main()
