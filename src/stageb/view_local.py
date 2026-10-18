"""Desktop Python viewer for actual Stage-B fields, never a cloud rendering."""
import argparse,json
from pathlib import Path
import numpy as np
from assembly import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,default=ROOT/'results/stage-b/cfd/reference-medium')
    p.add_argument('--paths',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:
        import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from stageb.rain import load_case,triangles
    centres,u,extra,meta,latest=load_case(a.case)
    fig=plt.figure(figsize=(12,7));ax=fig.add_subplot(111,projection='3d');fig.subplots_adjust(left=.13,right=.97)
    for name,color,alpha in [('building_vertices_m','silver',.20),('panel_vertices_m','teal',.8)]:
        ax.add_collection3d(Poly3DCollection(triangles(np.asarray(meta[name])),facecolors=color,edgecolors='black',linewidths=.35,alpha=alpha))
    pv=np.array(meta['panel_vertices_m']);target=pv.mean(0)
    # Zoom includes the upstream facade and panel; sampled arrows are real U.
    sel=np.all(np.abs(centres-target)<[10,8,10],axis=1)&(centres[:,2]>=0)
    ids=np.flatnonzero(sel)
    if len(ids)>1600:ids=np.random.default_rng(7).choice(ids,1600,replace=False)
    speed=np.linalg.norm(u[ids],axis=1);cloud=ax.scatter(*centres[ids].T,c=speed,s=5,cmap='viridis',alpha=.65)
    arrow=ids[::max(1,len(ids)//180)];ax.quiver(*centres[arrow].T,*u[arrow].T,length=.22,normalize=False,alpha=.35,linewidth=.5)
    colorbar=fig.colorbar(cloud,ax=ax,shrink=.65,pad=.04);colorbar.set_label('Speed [m/s]')
    if a.paths:
        paths=np.load(a.paths,allow_pickle=False)
        for name in paths.files:
            for curve in np.moveaxis(paths[name],1,0):ax.plot(*curve.T,color='royalblue',alpha=.3,linewidth=.6)
    ax.set(xlim=(target[0]-10,target[0]+10),ylim=(target[1]-8,target[1]+8),zlim=(0,20),xlabel='X downwind [m]',ylabel='Y lateral [m]',zlabel='Z up [m]')
    ax.set_box_aspect([20,16,20]);ax.set_title(f'Executed OpenFOAM {meta["release"]} | {len(centres):,} cells\nExploratory RANS; transfer coefficients and yield remain unqualified')
    controls=RadioButtons(fig.add_axes([.01,.72,.10,.16]),['Speed','T [C]','Yv [kg/kg]'])
    def update(label):
        values=speed if label=='Speed' else extra['T'][ids]-273.15 if label=='T [C]' else extra['Yv'][ids]
        cloud.set_array(values);cloud.set_clim(float(values.min()),float(values.max())+1e-15);colorbar.set_label(label);fig.canvas.draw_idle()
    controls.on_clicked(update)
    if a.check:
        for label in ['Speed','T [C]','Yv [kg/kg]']:update(label);fig.canvas.draw()
        plt.close(fig);print(json.dumps({'viewer':'native matplotlib','actual_CFD_cells':len(centres),'check':'passed','image_file_created':False}))
    else:plt.show()


if __name__=='__main__':main()
