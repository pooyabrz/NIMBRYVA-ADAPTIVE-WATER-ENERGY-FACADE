"""Tests of demonstrator logic, not validation of hardware or harvesting."""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from concept_demo import DemoController,settle,geometry_check

class ConceptDemoTests(unittest.TestCase):
    def collector(self):
        m=DemoController();m.request_face(True);settle(m);return m
    def test_full_turn_returns_to_periodic_zero_with_closed_ports(self):
        m=DemoController();self.assertTrue(m.request_move(360));m.check_invariants();settle(m)
        self.assertEqual(m.theta,360);self.assertEqual(m.theta%360,0)
    def test_water_input_outside_collection_pose_is_bypassed(self):
        m=DemoController();m.add_water(1);self.assertEqual(m.volume,0);self.assertEqual(m.bypass,1)
    def test_automatic_four_liter_cycle_conserves_water(self):
        m=self.collector()
        for _ in range(4):m.add_water()
        self.assertEqual(m.state,'CONNECTING');settle(m)
        self.assertAlmostEqual(m.volume,.2);self.assertAlmostEqual(m.storage,3.8);self.assertFalse(m.dock_engaged)
    def test_large_move_drains_before_moving(self):
        m=self.collector();m.add_water(2);self.assertTrue(m.request_move(m.theta+360));self.assertEqual(m.state,'CONNECTING')
        settle(m);self.assertAlmostEqual(m.volume,.2);self.assertAlmostEqual(m.storage,1.8);self.assertEqual(m.theta%360,180)
    def test_wind_interlock_blocks_rotation(self):
        m=DemoController();m.toggle_wind();self.assertFalse(m.request_move(360));self.assertEqual(m.moves,0)
    def test_wind_change_stops_demo_and_closes_ports(self):
        m=DemoController();m.request_move(360);m.advance(1);m.toggle_wind();m.check_invariants();self.assertEqual(m.state,'IDLE');self.assertFalse(m.vent_open)
    def test_receiver_capacity_blocks_discharge_and_motion(self):
        m=self.collector();m.add_water(2);m.storage=19;m.input_total+=19
        self.assertFalse(m.request_move(m.theta+360));self.assertEqual(m.state,'IDLE');self.assertFalse(m.inlet_open)
    def test_arrival_event_does_not_repeat_while_parked(self):
        m=self.collector();m.add_water(2);m.request_drain();settle(m);before=m.discharge_cycles
        for _ in range(20):m.advance(.1)
        self.assertEqual(m.discharge_cycles,before)
    def test_negative_or_nan_input_rejected(self):
        m=self.collector()
        for v in [-1,float('nan')]:
            with self.assertRaises(ValueError):m.add_water(v)
        self.assertFalse(m.request_move(float('inf')))
    def test_high_high_level_and_transfer_bypass(self):
        m=self.collector();m.add_water(4.5);self.assertIn('4.5',m.event);m.add_water(1);settle(m)
        self.assertAlmostEqual(m.bypass,1);self.assertAlmostEqual(m.storage,4.3)
    def test_geometry_includes_buffer_and_dock_clearance(self):
        r=geometry_check();self.assertAlmostEqual(r['swept_radius_m'],math.hypot(1.115,.04));self.assertGreater(r['fixed_dock_gap_m'],0);self.assertGreater(r['minimum_wall_gap_m'],.28)

if __name__=='__main__':unittest.main()
