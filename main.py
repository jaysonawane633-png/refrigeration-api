
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from CoolProp.CoolProp import PropsSI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
import psychrolib

psychrolib.SetUnitSystem(psychrolib.SI)

app = FastAPI(title="Refrigeration Calculator API")

class CycleInput(BaseModel):
    refrigerant: str = "R134a"
    evaporator_temp_C: float = -5
    condenser_temp_C: float = 40
    superheat_K: float = 5
    subcooling_K: float = 3
    mass_flow_kg_s: float = 0.03

def chart_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.get("/")
def home():
    return {"status": "ok", "message": "Refrigeration API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/solve")
def solve(data: CycleInput):
    try:
        fluid = data.refrigerant
        te = data.evaporator_temp_C + 273.15
        tc = data.condenser_temp_C + 273.15

        if not (180 < te < tc < 430):
            raise HTTPException(
                status_code=400,
                detail="Check evaporator and condenser temperatures."
            )
        if data.mass_flow_kg_s <= 0:
            raise HTTPException(
                status_code=400,
                detail="Mass flow must be positive."
            )

        # Saturation pressures
        pe = PropsSI("P", "T", te, "Q", 1, fluid)
        pc = PropsSI("P", "T", tc, "Q", 0, fluid)

        # State 1: evaporator outlet / compressor inlet
        t1 = te + data.superheat_K
        p1 = pe
        h1 = PropsSI("H", "T", t1, "P", p1, fluid)
        s1 = PropsSI("S", "T", t1, "P", p1, fluid)

        # State 2: isentropic compression approximation
        h2s = PropsSI("H", "P", pc, "S", s1, fluid)
        h2 = h1 + (h2s - h1) / 0.75
        t2 = PropsSI("T", "P", pc, "H", h2, fluid)

        # State 3: condenser outlet / expansion valve inlet
        t3 = tc - data.subcooling_K
        h3 = PropsSI("H", "T", t3, "P", pc, fluid)

        # State 4: throttling, h4 = h3
        h4 = h3

        q_evap = h1 - h4
        w_comp = h2 - h1
        q_cond = h2 - h3

        if min(q_evap, w_comp, q_cond) <= 0:
            raise ValueError(
                "Invalid cycle: check operating temperatures and refrigerant."
            )

        m = data.mass_flow_kg_s
        cooling_kw = m * q_evap / 1000
        compressor_kw = m * w_comp / 1000
        condenser_kw = m * q_cond / 1000
        cop = q_evap / w_comp

        # P-h diagram (cycle points)
        enthalpy = np.array([h1, h2, h3, h4, h1]) / 1000
        pressure = np.array([p1, pc, pc, pe, pe]) / 1000

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(enthalpy, pressure, marker="o")
        for i, (x, y) in enumerate(zip(enthalpy[:-1], pressure[:-1]), 1):
            ax.annotate(str(i), (x, y))
        ax.set_yscale("log")
        ax.set_xlabel("Enthalpy (kJ/kg)")
        ax.set_ylabel("Pressure (kPa)")
        ax.set_title(f"P-h Cycle: {fluid}")
        ax.grid(True, which="both")
        ph_png = chart_base64(fig)

        # Psychrometric chart: illustrative air-side process
        tdb = np.linspace(0, 45, 100)
        p_atm = 101325
        fig, ax = plt.subplots(figsize=(8, 5))
        for rh in [0.2, 0.4, 0.6, 0.8, 1.0]:
            w = [
                0.621945 * (rh * PropsSI("P", "T", t + 273.15, "Q", 0, "Water"))
                / (p_atm - rh * PropsSI(
                    "P", "T", t + 273.15, "Q", 0, "Water"
                ))
                for t in tdb
            ]
            ax.plot(tdb, np.array(w) * 1000, label=f"RH {rh:.0%}")
        ax.set_xlabel("Dry-bulb temperature (°C)")
        ax.set_ylabel("Humidity ratio (g water/kg dry air)")
        ax.set_title("Psychrometric Chart (illustrative)")
        ax.set_ylim(bottom=0)
        ax.grid(True)
        ax.legend()
        psych_png = chart_base64(fig)

        return {
            "refrigerant": fluid,
            "evaporator_pressure_kPa": pe / 1000,
            "condenser_pressure_kPa": pc / 1000,
            "h1_kJ_kg": h1 / 1000,
            "h2_kJ_kg": h2 / 1000,
            "h3_kJ_kg": h3 / 1000,
            "h4_kJ_kg": h4 / 1000,
            "cop": cop,
            "cooling_capacity_kW": cooling_kw,
            "compressor_power_kW": compressor_kw,
            "condenser_heat_rejection_kW": condenser_kw,
            "ph_chart_base64": ph_png,
            "psychrometric_chart_base64": psych_png,
            "note": (
                "Preliminary cycle estimate. The psychrometric chart shows "
                "air-property curves, not a measured vehicle cabin process. "
                "Component design requires additional inputs."
            )
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
