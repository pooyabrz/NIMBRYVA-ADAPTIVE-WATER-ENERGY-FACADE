"""Numerical verification and data/causality checks, NOT empirical validation."""
import argparse
import importlib.metadata
import hashlib
import json
import math
import sys
import unittest
import numpy as np
from reference_checks import ReferenceChecks
from core import SIGMA
from geometry import panel_corners,panel_frame,view_factors,ray_category
from physics import PanelModel
from weather import ROOT,cities,load_weather,model_config


def forcing(**overrides):
    w={'air_C':25.,'rh_pct':50.,'pressure_Pa':101325.,'wind_m_s':0.,
       'wind_x':0.,'wind_z':0.,'rain_mm_h':0.,'snowfall_cm_h':0.,
       'lw_down_W_m2':SIGMA*298.15**4}
    w.update({f'poa_{i}':0. for i in range(5)});w.update(overrides)
    return w


class CoupledChecks(unittest.TestCase):
    def test_isothermal_dark_dry_equilibrium(self):
        p=PanelModel();p.reset(25)
        r=p.step_hour(forcing(),2)
        np.testing.assert_allclose(p.state.temperature_C,[25,25],atol=1e-9)
        self.assertEqual(r['pv_gross_Wh'],0);self.assertEqual(r['drain_L'],0)

    def test_rain_projection_and_film_mass_balance(self):
        p=PanelModel();p.reset(25)
        r=p.step_hour(forcing(rh_pct=100.,rain_mm_h=10.),2)
        self.assertAlmostEqual(r['rain_input_L'],10*.85*math.cos(math.radians(50)),places=9)
        self.assertAlmostEqual(r['mass_residual_kg'],0,places=9)
        self.assertGreater(r['drain_L'],0)

    def test_radiative_cooling_and_latent_feedback(self):
        wet=PanelModel();wet.reset(25)
        dry=PanelModel();dry.reset(25)
        rw=wet.step_hour(forcing(rh_pct=95.,lw_down_W_m2=270.),5)
        dry.step_hour(forcing(rh_pct=10.,lw_down_W_m2=270.),5)
        self.assertGreater(rw['condensed_L'],0)
        self.assertGreater(wet.state.temperature_C[1],dry.state.temperature_C[1])

    def test_evaporation_cannot_consume_absent_water(self):
        p=PanelModel();p.reset(25);p.state.film_kg[0]=.03
        r=p.step_hour(forcing(rh_pct=1.,**{f'poa_{i}':800. for i in range(5)}),2)
        self.assertLessEqual(r['evaporated_L'],.03+1e-12)
        self.assertTrue((p.state.film_kg>=0).all())
        self.assertLess(abs(r['mass_residual_kg']),1e-10)

    def test_switch_preserves_thermal_and_film_state(self):
        c=model_config();c.update(solar_absorptance=[0,0],longwave_emissivity=[0,0],
             inter_face_conductance_W_m2_K=0,back_conductance_W_m2_K=0,
             front_convection_intercept_W_m2_K=0,front_convection_wind_slope=0)
        p=PanelModel(c);p.reset(25);p.state.temperature_C[:]=[40,10];p.state.film_kg[:]=[.005,.006]
        p.step_hour(forcing(),7)
        np.testing.assert_allclose(p.state.temperature_C,[40,10])
        np.testing.assert_allclose(p.state.film_kg,[.005,.006])

    def test_battery_conservation_with_surplus_and_empty_battery(self):
        p=PanelModel();p.reset(25,battery_fraction=1.)
        r=p.step_hour(forcing(**{f'poa_{i}':1000. for i in range(5)}),2)
        self.assertGreater(r['surplus_Wh'],0)
        self.assertLess(abs(r['battery_residual_Wh']),1e-9)
        p.reset(25,battery_fraction=0.);r=p.step_hour(forcing(),5)
        self.assertEqual(r['action_blocked'],1)
        self.assertAlmostEqual(r['unserved_Wh'],.2,places=9)
        self.assertGreaterEqual(p.state.battery_Wh,-1e-12)
        self.assertLess(abs(r['battery_residual_Wh']),1e-9)

    def test_cold_and_snow_states_are_flagged(self):
        p=PanelModel();p.reset(-2)
        r=p.step_hour(forcing(air_C=-2.,lw_down_W_m2=SIGMA*271.15**4),2)
        self.assertFalse(r['water_supported']);self.assertTrue(r['frost_unsupported'])
        p.reset(25);r=p.step_hour(forcing(snowfall_cm_h=.1),2)
        self.assertFalse(r['water_supported']);self.assertTrue(r['snow_unsupported'])

    def test_heavy_rain_does_not_create_numerical_freezing(self):
        p=PanelModel();p.reset(27);p.state.temperature_C[:]=[40,15]
        r=p.step_hour(forcing(air_C=27.,rh_pct=95.,wind_m_s=10.,rain_mm_h=100.,
                              lw_down_W_m2=420.),5)
        self.assertTrue(r['water_supported'])
        self.assertGreater(float(p.state.temperature_C.min()),10)
        self.assertLess(r['effective_dt_s'],60.)

    def test_synthetic_time_step_convergence(self):
        results=[]
        for dt in [60,30,15]:
            c=model_config();c['integration_dt_s']=dt;p=PanelModel(c);p.reset(25)
            r=p.step_hour(forcing(rh_pct=90.,lw_down_W_m2=280.),5)
            results.append((p.state.temperature_C.copy(),r['condensed_L']))
        self.assertLess(float(np.max(np.abs(results[1][0]-results[2][0]))),.08)
        self.assertLess(abs(results[1][1]-results[2][1]),.0002)


class GeometryChecks(unittest.TestCase):
    def test_frame_and_area(self):
        for tilt in [0,20,50,80,90]:
            u,v,n=panel_frame(tilt)
            np.testing.assert_allclose(np.cross(u,v),n,atol=1e-12)
            q=panel_corners(tilt,1.,10.)
            self.assertAlmostEqual(float(np.linalg.norm(np.cross(q[1]-q[0],q[3]-q[0]))),1.)

    def test_view_factor_limits_and_closure(self):
        self.assertEqual(view_factors(0,10,1e8,30,20), (1.,0.,0.))
        self.assertGreater(view_factors(0)[2],0)  # facade blocks even an upward-facing panel
        for t in [20,50,80,90]:
            fs=view_factors(t)
            self.assertAlmostEqual(sum(fs),1.,places=12)
            self.assertTrue(all(0<=f<=1 for f in fs))
        # Negligible distant wall recovers free-hemisphere analytic factors.
        fs,fg,fw=view_factors(60,10,1e8,30,20)
        self.assertAlmostEqual(fs,.75,delta=.002)
        self.assertAlmostEqual(fg,.25,delta=.002)
        self.assertEqual(fw,0.)

    def test_first_intersections(self):
        result=ray_category([[0,1,0],[0,-1,0],[0,0,-1]],10,.7,30,20)
        np.testing.assert_array_equal(result,[0,1,2])


class WeatherChecks(unittest.TestCase):
    def test_complete_reference_year_and_provenance(self):
        import pandas as pd
        quality=[q for q in json.loads((ROOT/'results/weather-quality.json').read_text()) if q['year']==2024]
        self.assertEqual(len(quality),12)
        self.assertTrue(all(q['status']=='passed' and q['hours']==8784 for q in quality))
        for city in cities():
            frame=pd.read_csv(ROOT/'data/normalized'/f"{city['id']}-2024.csv.gz")
            self.assertFalse(frame.isna().any().any())
            self.assertEqual(len(frame),8784)
        for record in (ROOT/'data/raw').glob('*.provenance.json'):
            meta=json.loads(record.read_text())
            raw=record.with_name(record.name.replace('.provenance.json','.json'))
            self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(),meta['sha256'])


class RLChecks(unittest.TestCase):
    def test_gym_contract_and_future_information_boundary(self):
        from environment import FacadeEnv
        from gymnasium.utils.env_checker import check_env
        env=FacadeEnv(['singapore'],episode_hours=24)
        check_env(env,skip_render_check=True)
        a,info=env.reset(seed=7)
        b,_=env.reset(seed=7)
        np.testing.assert_array_equal(a,b)
        before=env._observation().copy();original=env.frame.copy()
        # A large alteration of ONLY future data must not alter today's input.
        env.frame.loc[env.frame.index[env.t+1:],'ghi_W_m2']=0.
        env.frame.loc[env.frame.index[env.t+1:],'air_C']=40.
        np.testing.assert_array_equal(before,env._observation())
        env.frame=original
        for _ in range(24):
            obs,reward,terminated,truncated,result=env.step(2)
            self.assertTrue(np.isfinite(obs).all())
            self.assertLess(abs(result['mass_residual_kg']),1e-8)
            self.assertLess(abs(result['battery_residual_Wh']),1e-8)
            if terminated or truncated:break


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rl',action='store_true');a=parser.parse_args()
    suite=unittest.TestSuite()
    for cls in [ReferenceChecks,CoupledChecks,GeometryChecks,WeatherChecks]+([RLChecks] if a.rl else []):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    packages=['numpy','scipy','pandas','matplotlib','pvlib']+(['gymnasium'] if a.rl else [])
    output={'status':'passed' if result.wasSuccessful() else 'failed','tests_run':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors),
            'scope':'Numerical/limit/data/causality verification only; not measured or high-fidelity device validation.',
            'trained_policy':False,'field_validation':False,'CFD_executed':False,
            'python':sys.version,'versions':{p:importlib.metadata.version(p) for p in packages}}
    (ROOT/'results'/('verification-rl.json' if a.rl else 'verification.json')).write_text(json.dumps(output,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
