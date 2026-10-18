"""Rebuild the active configuration from frozen evidence plus declared choices."""
import json,math
from assembly import ROOT,MODEL_VERSION,parameters


def build():
    p=parameters();c={
      'model_version':MODEL_VERSION,'status':'UNVALIDATED_CASSETTE_SCREENING_WITH_EVIDENCE_GAPS',
      'area_m2':p['panel_width_m']*p['rotation_chord_m'],'dew_area_m2':1.6,
      'panel_width_m':p['panel_width_m'],'rotation_chord_m':p['rotation_chord_m'],
      'tilts_deg':[20,35,50,65,80],'wall_standoff_m':1.,'mounting_height_m':8.,
      'building_height_m':p['building_height_m'],'building_width_m':p['building_length_m'],
      'facade_azimuth_north_cw_deg':180.,'facade_orientation':'absolute_south_common_reference',
      'ground_shortwave_albedo':.2,'pv_efficiency_at25':435/(1000*p['panel_width_m']*p['rotation_chord_m']),
      'pv_gamma_per_K':-.003,'pv_effective_absorptance':.9,'pv_longwave_emissivity':.84,
      'pv_solar_heat_cell_fraction':.95,'pv_capacity_multiplier':1.,
      'laminate_contact_R_m2_K_W':0.,'pv_xps_contact_R_m2_K_W':.01,
      'frame_cp_J_kg_K':900.,'frame_pv_contact_W_K':2.,'frame_ambient_W_K':.5,
      'bridge_paths':[
        {'name':'six protected PEEK spacers, proposed geometry','count':6,'area_each_m2':math.pi*.008**2/4,
         'segments':[{'length_m':.04,'k_W_m_K':.27}],'contact_R_m2_K_W':.002},
        {'name':'two stainless M4 parallel leakage paths, proposed geometry','count':2,'area_each_m2':math.pi*.004**2/4,
         'segments':[{'length_m':.04,'k_W_m_K':15.}],'contact_R_m2_K_W':.002}],
      'bridge_multiplier':1.,'cavity_gap_m':.02,'port_inlet_m2':.01134,'port_outlet_m2':.01134,
      'port_discharge_coefficient':.6,'port_delta_Cp':.2,'channel_h_multiplier':1.,
      'cavity_radiation_effective_epsilon':.8,'lewis_number':.86,
      'external_h_multiplier':1.,'natural_convection_floor_W_m2_K':2.5,
      'rain_interception_efficiency':.85,'drop_fall_speed_m_s':6.,
      'jump_fraction':.8,'collector_efficiency':.85,'coating_condition':1.,
      'droplet_footprints_m2':1e6,'outer_adv_deg':100.,'outer_rec_deg':60.,
      'water_viscosity_Pa_s':.001,
      'battery_capacity_Wh':200.,'charge_efficiency':.95,'discharge_efficiency':.95,'dc_efficiency':.95,
      'idle_W':.5,'motor_running_W':30.,'motor_running_upper_W':72.,
      'motor_energy_policy':'upper_running','motor_output_rpm':.065,'motor_ramp_s':5.,
      'motor_rated_output_Nm':600.,'motor_holding_Nm':3000.,'motor_tilting_Nm':1500.,
      'motor_radial_N':16000.,'motor_min_output_deg':0.,'motor_max_output_deg':360.,
      'motor_accuracy_deg':.1,'motor_temperature_min_C':-30.,'motor_temperature_max_C':80.,
      'pivot_CG_offset_m':.002,'bearing_friction_Nm':.3,'wind_force_coefficient':1.5,
      'wind_CP_offset_fraction':.25,'wind_gust_factor':1.5,'wind_mount_lever_m':.1,
      'torque_screening_factor':2.,'clearance_allowance_m':.05,
      'terrain_alpha':.22,'local_lapse_K_m':-.0065,
      'spectral_points':512,'dew_unknown_tail_epsilon':.9,'dew_angular_multiplier':1.,
      'sky_temperature_cap_offset_K':10.,'sky_spectral_policy':'nominal',
      'wall_temperature_offset_K':0.,'integration_dt_s':60.,'nonlinear_tolerance_K':1e-7,
      'geometry_angle_step_deg':2.,'geometry_rays':1024,'geometry_area_samples':3,
      'reward_water_per_L':1.,'reward_unserved_per_Wh':1.,'reward_switch_penalty':.002,
      'reward_final_SOC_deficit_per_Wh':.1,
      'evidence_policy':'All interval/sensitivity values are epistemic screening choices, not confidence intervals or certified ratings.'}
    return c


if __name__=='__main__':
    dest=ROOT/'config/model.json';dest.write_text(json.dumps(build(),indent=2)+'\n')
    print(dest)
