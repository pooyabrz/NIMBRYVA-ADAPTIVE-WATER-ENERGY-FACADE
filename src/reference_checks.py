"""Run phase-1 numerical/physical limiting-case checks; not device validation."""
import json
import math
import unittest
from pathlib import Path
from core import (SIGMA, saturation_pressure_pa, dew_point_C, standard_pressure_pa,
                  active_normal, ideal_rain_capture_L_h, net_longwave_loss_W_m2,
                  condensation_reference_kg_m2_s, pv_reference_W)


class ReferenceChecks(unittest.TestCase):
    def test_saturation_at_20_C(self):
        self.assertAlmostEqual(saturation_pressure_pa(20),2338.8,delta=1.0)

    def test_saturation_at_100_C(self):
        self.assertAlmostEqual(saturation_pressure_pa(100),101418,delta=20)

    def test_dew_point_and_saturation_limit(self):
        self.assertAlmostEqual(dew_point_C(30,.6),21.4,delta=.1)
        self.assertAlmostEqual(dew_point_C(20,1),20,places=8)

    def test_dew_point_inversion_across_domain(self):
        for t in [0,5,20,35,50]:
            for rh in [.2,.5,.9,1.0]:
                td=dew_point_C(t,rh)
                self.assertLessEqual(td,t+1e-9)
                self.assertAlmostEqual(saturation_pressure_pa(td)/saturation_pressure_pa(t),rh,places=9)

    def test_bad_relative_humidity_rejected(self):
        for rh in [-.1,0,60,float('nan')]:
            with self.assertRaises(ValueError):
                dew_point_C(30,rh)

    def test_pressure_reference(self):
        self.assertEqual(standard_pressure_pa(0),101325)
        self.assertAlmostEqual(standard_pressure_pa(1000),89875,delta=10)

    def test_rain_no_wind_projection(self):
        for beta in [0,30,60,90]:
            got=ideal_rain_capture_L_h(1,10,(0,-6,0),active_normal(beta))
            self.assertAlmostEqual(got,10*math.cos(math.radians(beta)),places=8)

    def test_rain_direction_and_units(self):
        self.assertAlmostEqual(ideal_rain_capture_L_h(2,10,(0,-6,-3),active_normal(90)),10)
        self.assertEqual(ideal_rain_capture_L_h(1,10,(0,-6,3),active_normal(90)),0)
        self.assertEqual(ideal_rain_capture_L_h(1,0,(0,-6,-3),active_normal(45)),0)

    def test_radiative_equilibrium(self):
        temp=20
        flux=net_longwave_loss_W_m2(temp,.95,SIGMA*(temp+273.15)**4,temp,temp,.6,.1,.3)
        self.assertAlmostEqual(flux,0,places=9)

    def test_clearer_sky_releases_more_heat(self):
        clear=net_longwave_loss_W_m2(20,.95,280,20,20,.8,.1,.1)
        cloudy=net_longwave_loss_W_m2(20,.95,380,20,20,.8,.1,.1)
        self.assertGreater(clear,cloudy)

    def test_condensation_gate(self):
        self.assertEqual(condensation_reference_kg_m2_s(30,.6,25,.01),0)
        self.assertGreater(condensation_reference_kg_m2_s(30,.6,20,.01),0)
        self.assertEqual(condensation_reference_kg_m2_s(30,.6,20,0),0)

    def test_pv_darkness_reference_and_temperature(self):
        self.assertEqual(pv_reference_W(1,0,.2,-.004,25),0)
        self.assertAlmostEqual(pv_reference_W(1,1000,.2,-.004,25),200)
        self.assertLess(pv_reference_W(1,1000,.2,-.004,50),200)

    def test_invalid_view_factors(self):
        with self.assertRaises(ValueError):
            net_longwave_loss_W_m2(20,.95,300,20,20,.8,.3,.1)


