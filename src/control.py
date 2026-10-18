"""Causal controller: only the last available interval is inspected."""


def fixed_pv(last_weather, model):
    return min(range(len(model.c['tilts_deg'])), key=lambda i:abs(model.c['tilts_deg'][i]-50))


def causal_rule(last_weather, model):
    tilts = model.c['tilts_deg']; nt = len(tilts)
    # Thresholds are transparent engineering choices, not optimized claims.
    if last_weather['rain_mm_h']>.1:
        tilt = min(range(nt),key=lambda i:abs(tilts[i]-35))
        face = model.state.face  # avoid spending energy on a cosmetic rain switch
    elif last_weather['ghi_W_m2']>80:
        face = 0
        tilt = max(range(nt),key=lambda i:last_weather[f'poa_{i}'])
    else:
        face = 1; tilt = 0  # sky view for night-time radiative cooling
    return face*nt+tilt
