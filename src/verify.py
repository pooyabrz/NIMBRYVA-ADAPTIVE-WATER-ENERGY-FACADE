"""Conservation, limiting cases, convergence and interface verification.

These tests verify equations/software. No test asserts field accuracy.
"""
import argparse,json,math,tempfile,unittest,time
from pathlib import Path
import numpy as np
from core import SIGMA,saturation_pressure_pa,dew_point_C
from assembly import ROOT,MODEL_VERSION,compile_assembly,bridge_conductance,validate_config
from weather import model_config
from spectral import SpectralRadiation,planck_exitance_um,validate_optics,SpectralForcing
from physics import PanelModel
from flow import mixing_ratio,channel_flow,cavity_balance
from actuator import transition,motion_angle,load_envelope,allowable_wind
from drainage import drain


def forcing(**override):
    w={'air_C':25.,'rh_pct':80.,'pressure_Pa':101325.,'wind_m_s':2.,'wind_x':0.,'wind_z':-2.,
       'lw_down_W_m2':320.,'dni_W_m2':0.,'dhi_W_m2':0.,'ghi_W_m2':0.,'rain_mm_h':0.,
       'sun_x':0.,'sun_y':-1.,'sun_z':0.,'snowfall_cm_h':0.,'dew_C':21.3094,'cloud_pct':0.}
    w.update(override);return w


class CassetteChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=model_config();cls.assembly=compile_assembly(cls.cfg);cls.rad=SpectralRadiation(cls.cfg)

    def test_01_layer_storage_and_internal_closure(self):
        a=self.assembly
        self.assertEqual(len(a['layers']),8);self.assertTrue((a['capacity_J_K']>0).all())
        np.testing.assert_allclose(a['laplacian_W_K'].sum(1),0,atol=1e-10)
        np.testing.assert_allclose(a['laplacian_W_K'],a['laplacian_W_K'].T)

    def test_02_bridge_series_parallel_analytic(self):
        paths=[{'name':'test','count':2,'area_each_m2':.001,'segments':[{'length_m':.1,'k_W_m_K':10}], 'contact_R_m2_K_W':.01}]
        g,_=bridge_conductance(paths);self.assertAlmostEqual(g,.1)

    def test_03_cannot_use_old_configuration(self):
        c=dict(self.cfg);c.pop('model_version')
        with self.assertRaises(ValueError):validate_config(c)

    def test_04_full_sweep_wall_collision(self):
        c=dict(self.cfg);c['wall_standoff_m']=.7
        with self.assertRaises(ValueError):validate_config(c)

    def test_05_blackbody_spectral_integral(self):
        x=np.geomspace(.1,1e5,20000)
        for t in [0,25,60]:
            integral=np.trapezoid(planck_exitance_um(x,t),x)
            self.assertLess(abs(integral/(SIGMA*(t+273.15)**4)-1),1e-6)

    def test_06_raw_unphysical_data_are_rejected(self):
        r=self.rad.raw;self.assertEqual(int((r['absorptance_raw']>1).sum()),13)
        with self.assertRaises(ValueError):validate_optics(r['wavelength_um'],r['absorptance_raw'])

    def test_07_trusted_window_matches_published_average(self):
        r=self.rad.raw;mask=(r['wavelength_um']>=8)&(r['wavelength_um']<=13)
        x=r['wavelength_um'][mask];y=r['absorptance_raw'][mask]
        avg=np.trapezoid(y,x)/(x[-1]-x[0]);self.assertAlmostEqual(avg,.935,places=3)

    def test_08_sky_spectral_identifiability_envelope(self):
        b=self.rad.sky(25,350)
        self.assertLess(b['low'],b['nominal']);self.assertLess(b['nominal'],b['high'])
        self.assertGreater(b['high']-b['low'],1);self.assertFalse(b['shape_identified'])

    def test_09_closed_ports_cannot_provide_unlimited_vapour(self):
        c=dict(self.cfg);c['port_inlet_m2']=0
        wa=mixing_ratio(25,.8,101325);f=channel_flow(c,200,2,25,22,101325,wa)
        r=cavity_balance(f['dry_air_kg_s'],f['h_W_m2_K'],1.6,25,24,18,wa,101325,0,60)
        self.assertEqual(r['mass_kg_s'],0)

    def test_10_cavity_moist_mass_enthalpy_and_supply_bound(self):
        wa=mixing_ratio(25,.8,101325)
        for mda in [1e-8,1e-4,.01]:
            r=cavity_balance(mda,5,1.6,25,26,18,wa,101325,0,60)
            self.assertGreaterEqual(r['mass_kg_s'],0);self.assertLessEqual(r['mass_kg_s'],mda*wa)
            self.assertLess(abs(r['humidity_balance_kg_s']),1e-14);self.assertLess(abs(r['enthalpy_residual_W']),1e-9)

    def test_11_evaporation_is_inventory_limited(self):
        wa=mixing_ratio(25,.1,101325)
        r=cavity_balance(.01,5,1.6,25,28,28,wa,101325,1e-6,60)
        self.assertGreaterEqual(r['mass_kg_s'],-1e-6/60-1e-14)

    def test_12_fixed_T_RH_pressure_isolation(self):
        ta=25.;rh=.8;partial=rh*saturation_pressure_pa(ta)
        rho_v=[partial/(461.52*(ta+273.15)) for p in [101325,61640]]
        self.assertEqual(rho_v[0],rho_v[1]);self.assertAlmostEqual(dew_point_C(ta,rh),21.3094,places=4)
        self.assertGreater(mixing_ratio(ta,rh,61640),mixing_ratio(ta,rh,101325))

    def test_13_finite_flip_uses_one_motor_reference(self):
        m=transition(self.cfg,50,230)
        self.assertAlmostEqual(m['duration_s'],466.53846153846)
        self.assertAlmostEqual(m['running_upper_Wh'],9.33076923077)
        self.assertAlmostEqual(motion_angle(m,m['duration_s']/2),140)
        self.assertEqual(motion_angle(m,1e6),230)

    def test_14_dynamic_wind_screen_and_holding_distinct(self):
        lo=load_envelope(self.cfg,self.assembly,5);hi=load_envelope(self.cfg,self.assembly,20)
        self.assertTrue(lo['screen_pass']);self.assertFalse(hi['screen_pass'])
        self.assertGreater(hi['drive_utilization'],hi['hold_utilization'])
        self.assertAlmostEqual(hi['force_N']/lo['force_N'],16)

    def test_15_low_battery_rejects_motion_without_teleporting(self):
        m=PanelModel();m.reset(25,battery_fraction=0)
        r=m.step_hour(forcing(),7)
        self.assertEqual(r['action_blocked'],1);self.assertEqual(m.state.angle_deg,50)

    def test_16_wind_veto(self):
        m=PanelModel();m.reset(25)
        r=m.step_hour(forcing(wind_m_s=20),7)
        self.assertEqual(r['action_blocked'],1);self.assertEqual(r['motor_Wh'],0)

    def test_17_three_reservoir_mass_battery_heat_closure(self):
        m=PanelModel();m.reset(25)
        for w,action in [(forcing(),7),(forcing(rain_mm_h=35,ghi_W_m2=300,dhi_W_m2=300),2),(forcing(rh_pct=20),2)]:
            r=m.step_hour(w,action)
            self.assertLess(abs(r['mass_residual_kg']),1e-10);self.assertLess(abs(r['battery_residual_Wh']),1e-9)
            self.assertLess(r['max_thermal_residual_J'],.005);self.assertLess(r['max_cavity_enthalpy_residual_W'],1e-8)
            self.assertTrue((m.state.film_kg>=0).all());self.assertGreaterEqual(m.state.battery_Wh,-1e-9)

    def test_18_motor_energy_integrated_during_rotation(self):
        m=PanelModel();m.reset(25);r=m.step_hour(forcing(),7)
        self.assertAlmostEqual(r['motor_Wh'],9.33076923077,places=9)
        self.assertAlmostEqual(r['motion_seconds'],466.53846153846,places=8)

    def test_19_freezing_and_snow_remain_unsupported(self):
        m=PanelModel();m.reset(-3)
        r=m.step_hour(forcing(air_C=-3,rh_pct=80,lw_down_W_m2=260,snowfall_cm_h=.1),2)
        self.assertEqual(r['water_supported'],0);self.assertEqual(r['frost_unsupported'],1);self.assertEqual(r['snow_unsupported'],1)

    def test_20_spectrum_does_not_imply_validation(self):
        self.assertGreater(self.rad.audit()['blackbody_fraction_above_trusted_20um'],.25)
        m=PanelModel();m.reset(25);r=m.step_hour(forcing(),2)
        self.assertFalse(r['physical_yield_claim_supported'])

    def test_21_drainage_mass_and_degraded_condition(self):
        c=dict(self.cfg)
        f,d,l,detail=drain(c,.01,.01,1.6,-1,200,20,60,cnf=True)
        self.assertAlmostEqual(f+d+l,.01);self.assertGreater(detail['jumping_kg'],0)
        c['coating_condition']=0
        _,_,_,detail=drain(c,.01,.01,1.6,-1,200,20,60,cnf=True)
        self.assertEqual(detail['jumping_kg'],0)

    def test_22_time_step_convergence(self):
        records=[]
        for dt in [60,30,15]:
            c=dict(self.cfg);c['integration_dt_s']=dt;m=PanelModel(c);m.reset(25)
            drain_total=0
            for hour in range(3):
                r=m.step_hour(forcing(),7);drain_total+=r['drain_L']
            records.append({'dt_s':dt,'dew_C':r['surface_dew_C'],'drain_L':drain_total})
        d0=abs(records[0]['dew_C']-records[1]['dew_C']);d1=abs(records[1]['dew_C']-records[2]['dew_C'])
        self.assertLess(d1,d0*1.1+1e-6);self.assertLess(d1,.1)
        self.convergence=records
        out=ROOT/'results/cassette-v2';out.mkdir(parents=True,exist_ok=True)
        (out/'timestep-convergence.json').write_text(json.dumps(records,indent=2))

    def test_23_rl_old_policy_rejected_before_model_loading(self):
        from evaluate_rl import validate_policy_metadata
        with self.assertRaisesRegex(ValueError,'Legacy policy'):
            validate_policy_metadata(ROOT/'legacy_v1/rl/ppo-seed7-2024/policy.zip',self.cfg)

    def test_24_invalid_forcing_rejected(self):
        m=PanelModel();m.reset(25)
        with self.assertRaises(ValueError):m.step_hour(forcing(rain_mm_h=-1),2)
        with self.assertRaises(ValueError):m.step_hour(forcing(lw_down_W_m2=float('nan')),2)

    def test_25_policy_state_and_causal_mask(self):
        from environment import FacadeEnv
        env=FacadeEnv(['singapore'],episode_hours=3);obs,_=env.reset(seed=7)
        self.assertEqual(obs.shape,(48,));self.assertTrue(env.observation_space.contains(obs))
        saved=obs.copy();env.frame=env.frame.copy();env.frame.iloc[env.t+1,env.frame.columns.get_loc('rh_pct')]=10
        np.testing.assert_array_equal(env._observation(),saved)
        nxt,_,_,_,info=env.step(2);self.assertEqual(nxt.shape,(48,));self.assertEqual(info['model_version'],MODEL_VERSION)

    def test_26_spectral_grid_convergence(self):
        c=dict(self.cfg);c['spectral_points']=1024;r=SpectralRadiation(c)
        self.assertLess(abs(r.exitance(25)-self.rad.exitance(25)),.05)

    def test_27_bridge_is_not_old_unmeasured_constant(self):
        self.assertAlmostEqual(self.assembly['bridge_W_K'],.007394222866959245)
        self.assertNotAlmostEqual(self.assembly['bridge_W_K'],.1)

    def test_28_signed_channel_flow_and_zero_pressure(self):
        wa=mixing_ratio(25,.8,101325);c=dict(self.cfg);c['port_delta_Cp']=0
        zero=channel_flow(c,50,0,25,25,101325,wa);self.assertEqual(zero['volume_m3_s'],0)
        plus=channel_flow(c,50,0,25,30,101325,wa);minus=channel_flow(c,230,0,25,30,101325,wa)
        self.assertAlmostEqual(plus['volume_m3_s'],-minus['volume_m3_s'])

    def test_29_external_spectral_units_time_and_energy_gate(self):
        import pandas as pd,hashlib
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'spectrum.npz';sw=np.linspace(.28,4,400);lw=np.geomspace(2.5,1000,700)
            direct=np.full(len(sw),100/(sw[-1]-sw[0]));diffuse=direct/2
            sky=planck_exitance_um(lw,25);sky*=350/np.trapezoid(sky,lw)
            np.savez_compressed(p,times_utc=np.array(['2024-07-15T00:00:00Z']),sw_um=sw,lw_um=lw,
              dni_W_m2_um=direct[None,:],dhi_W_m2_um=diffuse[None,:],sky_W_m2_um=sky[None,:])
            meta={'source':'manufactured verification-only spectral boundary',
              'units':{'wavelength':'um','spectral_irradiance':'W m-2 um-1'},
              'profile_sha256':hashlib.sha256(b'manufactured-test').hexdigest(),'model_version':'analytic-test','purpose':'verification_only'}
            p.with_suffix('.json').write_text(json.dumps(meta));s=SpectralForcing(p)
            b=s.boundary(pd.Timestamp('2024-07-15T00:00Z'),forcing(dni_W_m2=100,dhi_W_m2=50,lw_down_W_m2=350),self.rad)
            self.assertTrue(b['validation_only'])
            with self.assertRaisesRegex(ValueError,'disagreement'):
                s.boundary(pd.Timestamp('2024-07-15T00:00Z'),forcing(dni_W_m2=300,dhi_W_m2=50,lw_down_W_m2=350),self.rad)
            with self.assertRaisesRegex(ValueError,'No external spectrum'):
                s.boundary(pd.Timestamp('2024-07-16T00:00Z'),forcing(),self.rad)
            meta['units']['wavelength']='nm';p.with_suffix('.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'units'):SpectralForcing(p)

    def test_30_geometry_area_ray_resolution_convergence(self):
        from geometry import cassette_view_table
        c=self.cfg;d=self.assembly['assembly_depth_m'];values=[]
        for samples,rays in [(3,1024),(5,4096),(7,8192)]:
            angles,v=cassette_view_table(c['panel_width_m'],c['rotation_chord_m'],8.,1.,c['building_height_m'],c['building_width_m'],d,10,rays,samples)
            self.assertLess(abs(v.sum(2)-1).max(),1e-10)
            values.append(v[[2,5,8,20,23,26]])
        self.assertLess(np.max(np.abs(values[-1]-values[-2])),.01)
        out=ROOT/'results/cassette-v2';out.mkdir(parents=True,exist_ok=True)
        (out/'geometry-convergence.json').write_text(json.dumps({'angles_deg':[20,50,80,200,230,260],
          'coarse_to_fine_max_absolute_change':float(np.max(np.abs(values[0]-values[2]))),
          'medium_to_fine_max_absolute_change':float(np.max(np.abs(values[1]-values[2]))),
          'scope':'area and ray quadrature of ideal isolated flat panels only'},indent=2))

    def test_31_cavity_counterflow_root_and_wet_gravity(self):
        from flow import coupled_cavity
        w=mixing_ratio(25,.8,101325)
        f,b=coupled_cavity(self.cfg,230,2,25,26,20,w,101325,.1,60)
        f2=channel_flow(self.cfg,230,2,25,b['temperature_C'],101325,w)
        self.assertAlmostEqual(f['volume_m3_s'],f2['volume_m3_s'],places=8)
        dry=load_envelope(self.cfg,self.assembly,5,film_mass=0)
        wet=load_envelope(self.cfg,self.assembly,5,film_mass=1)
        self.assertGreater(wet['required_drive_Nm']-dry['required_drive_Nm'],10)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true',help='Reserved; all conservation checks always run');parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CassetteChecks);start=time.time()
    r=unittest.TextTestRunner(verbosity=2).run(suite)
    out=ROOT/'results/cassette-v2';out.mkdir(parents=True,exist_ok=True)
    (out/'verification.json').write_text(json.dumps({'model_version':MODEL_VERSION,'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'passed':r.wasSuccessful(),'seconds':time.time()-start,'empirical_validation_performed':False},indent=2))
    raise SystemExit(0 if r.wasSuccessful() else 1)


if __name__=='__main__':main()
