from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mne
import numpy as np
import tempfile
import os
import httpx
from pydantic import BaseModel
from pydantic import BaseModel as BM
from typing import Optional, List

app = FastAPI(
    title="NeuroKine EEG Analysis API",
    max_upload_size=100 * 1024 * 1024  # 100MB
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://neurokine-emr.vercel.app",
        "http://localhost:5173",
        "http://localhost:5174",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

class BandPower(BaseModel):
    delta: float
    theta: float
    alpha: float
    beta: float
    gamma: float

class ChannelAnalysis(BaseModel):
    channel: str
    dominant_freq: float
    dominant_band: str
    alpha_power: float
    theta_power: float
    delta_power: float

class EEGAnalysis(BaseModel):
    duration_seconds: float
    num_channels: int
    sampling_rate: float
    channel_names: List[str]
    global_band_powers: BandPower
    posterior_dominant_rhythm: Optional[float]
    posterior_dominant_amplitude: Optional[float]
    interhemispheric_asymmetry: Optional[float]
    background_classification: str
    abnormalities: List[str]
    channel_analyses: List[ChannelAnalysis]
    ilae_background_grade: int
    summary: str

def compute_band_power(data, sfreq, band):
    from scipy.signal import welch
    lo, hi = band
    freqs, psd = welch(data, sfreq, nperseg=min(int(sfreq*2), len(data)))
    idx = np.logical_and(freqs >= lo, freqs <= hi)
    return float(np.mean(psd[idx]))

def classify_background(alpha_rel, theta_rel, delta_rel, pdr):
    abnormalities = []
    if pdr is not None:
        if pdr < 8:
            abnormalities.append(f"Slowed posterior dominant rhythm ({pdr:.1f} Hz, normal ≥8 Hz in adults)")
        elif pdr > 12:
            abnormalities.append(f"Fast posterior dominant rhythm ({pdr:.1f} Hz)")
    if delta_rel > 0.3:
        abnormalities.append(f"Excess delta activity ({delta_rel*100:.0f}% of power)")
    if theta_rel > 0.4:
        abnormalities.append(f"Excess theta activity ({theta_rel*100:.0f}% of power)")
    if alpha_rel < 0.1 and pdr is not None:
        abnormalities.append("Reduced alpha rhythm")
    return abnormalities

def get_ilae_grade(pdr, delta_rel, theta_rel, alpha_rel):
    if pdr is not None and pdr >= 8 and delta_rel < 0.1 and theta_rel < 0.2:
        return 1
    elif pdr is not None and pdr >= 6:
        return 2
    elif theta_rel > 0.5 or delta_rel > 0.3:
        return 3
    elif delta_rel > 0.5:
        return 4
    else:
        return 2

@app.get("/")
def root():
    return {"status": "NeuroKine EEG Analysis API", "version": "1.0"}

SUPABASE_URL = "https://yaihdkwgovrgbnfrrqjf.supabase.co"

class StaffUserCreate(BM):
    email: str
    password: str
    name: str
    staff_id: str

class StaffUserUpdate(BM):
    staff_id: str
    password: str

@app.post("/auth/create-staff")
async def create_staff_user(payload: StaffUserCreate):
    service_key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not service_key:
        raise HTTPException(500, "Service key not configured")

    async with httpx.AsyncClient() as client:
        # Create user in Supabase Auth
        res = await client.post(
            f"{SUPABASE_URL}/auth/v1/admin/users",
            headers={
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}",
                "Content-Type": "application/json"
            },
            json={
                "email": payload.email,
                "password": payload.password,
                "email_confirm": True,
                "user_metadata": {
                    "name": payload.name,
                    "role": "staff",
                    "staff_id": payload.staff_id
                }
            }
        )

        if res.status_code not in [200, 201]:
            error = res.json()
            raise HTTPException(
                res.status_code,
                error.get("message", "Failed to create user")
            )

        user_data = res.json()
        user_id = user_data.get("id")

        # Update nk_staff with supabase_user_id
        await client.patch(
            f"{SUPABASE_URL}/rest/v1/nk_staff?id=eq.{payload.staff_id}",
            headers={
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}",
                "Content-Type": "application/json"
            },
            json={"supabase_user_id": user_id}
        )

        # Try updating nk_doctors too (in case it's a doctor account)
        await client.patch(
            f"{SUPABASE_URL}/rest/v1/nk_doctors?id=eq.{payload.staff_id}",
            headers={
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}",
                "Content-Type": "application/json"
            },
            json={"supabase_user_id": user_id}
        )

        return {"success": True, "user_id": user_id}

@app.post("/auth/update-staff-password")
async def update_staff_password(payload: StaffUserUpdate):
    service_key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not service_key:
        raise HTTPException(500, "Service key not configured")

    async with httpx.AsyncClient() as client:
        staff_res = await client.get(
            f"{SUPABASE_URL}/rest/v1/nk_staff?id=eq.{payload.staff_id}&select=supabase_user_id",
            headers={
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}"
            }
        )
        staff = staff_res.json()
        if not staff or not staff[0].get("supabase_user_id"):
            raise HTTPException(404, "Staff user not found")

        user_id = staff[0]["supabase_user_id"]

        res = await client.put(
            f"{SUPABASE_URL}/auth/v1/admin/users/{user_id}",
            headers={
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}",
                "Content-Type": "application/json"
            },
            json={"password": payload.password}
        )

        if res.status_code not in [200, 201]:
            raise HTTPException(res.status_code, "Failed to update password")

        return {"success": True}

@app.delete("/auth/delete-staff/{staff_id}")
async def delete_staff_user(staff_id: str):
    service_key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not service_key:
        raise HTTPException(500, "Service key not configured")

    async with httpx.AsyncClient() as client:
        staff_res = await client.get(
            f"{SUPABASE_URL}/rest/v1/nk_staff?id=eq.{staff_id}&select=supabase_user_id",
            headers={
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}"
            }
        )
        staff = staff_res.json()
        if not staff or not staff[0].get("supabase_user_id"):
            return {"success": True, "note": "No auth user to delete"}

        user_id = staff[0]["supabase_user_id"]

        await client.delete(
            f"{SUPABASE_URL}/auth/v1/admin/users/{user_id}",
            headers={
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}"
            }
        )

        return {"success": True}

@app.post("/analyze", response_model=EEGAnalysis)
async def analyze_eeg(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.edf', '.bdf')):
        raise HTTPException(400, "Only EDF and BDF files supported")

    # Read content first to check size
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    print(f"Received file: {file.filename}, size: {file_size_mb:.1f}MB")

    if file_size_mb > 100:
        raise HTTPException(413, f"File too large ({file_size_mb:.0f}MB). Maximum 100MB.")

    with tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(file.filename)[1],
        delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        
        sfreq = raw.info['sfreq']
        duration = raw.times[-1]
        ch_names = raw.ch_names
        data = raw.get_data()
        
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        global_powers = {}
        total_power = 0
        for band_name, band_range in bands.items():
            bp = np.mean([
                compute_band_power(data[i], sfreq, band_range)
                for i in range(len(ch_names))
            ])
            global_powers[band_name] = bp
            total_power += bp
        
        rel_powers = {k: v/total_power for k, v in global_powers.items()}
        
        posterior_channels = [
            i for i, ch in enumerate(ch_names)
            if any(x in ch.upper() for x in ['O1','O2','OZ','P3','P4','PZ','P7','P8'])
        ]
        
        pdr = None
        pdr_amp = None
        if posterior_channels:
            from scipy.signal import welch
            post_data = np.mean(data[posterior_channels], axis=0)
            freqs, psd = welch(post_data, sfreq, nperseg=min(int(sfreq*4), len(post_data)))
            alpha_mask = np.logical_and(freqs >= 8, freqs <= 13)
            if alpha_mask.any():
                pdr = float(freqs[alpha_mask][np.argmax(psd[alpha_mask])])
                pdr_amp = float(np.sqrt(np.max(psd[alpha_mask])) * 1e6)
        
        left_ch = [i for i, ch in enumerate(ch_names)
                   if any(x in ch.upper() for x in ['F3','C3','P3','O1','T3','T5'])]
        right_ch = [i for i, ch in enumerate(ch_names)
                    if any(x in ch.upper() for x in ['F4','C4','P4','O2','T4','T6'])]
        
        asymmetry = None
        if left_ch and right_ch:
            left_alpha = np.mean([compute_band_power(data[i], sfreq, (8,13)) for i in left_ch])
            right_alpha = np.mean([compute_band_power(data[i], sfreq, (8,13)) for i in right_ch])
            if left_alpha + right_alpha > 0:
                asymmetry = float(abs(left_alpha - right_alpha) / (left_alpha + right_alpha) * 100)
        
        channel_analyses = []
        for i, ch in enumerate(ch_names[:32]):
            ch_powers = {}
            ch_total = 0
            for bn, br in bands.items():
                bp = compute_band_power(data[i], sfreq, br)
                ch_powers[bn] = bp
                ch_total += bp
            dom_band = max(ch_powers, key=ch_powers.get)
            from scipy.signal import welch
            freqs_ch, psd_ch = welch(data[i], sfreq, nperseg=min(int(sfreq*2), len(data[i])))
            dom_freq = float(freqs_ch[np.argmax(psd_ch[freqs_ch <= 50])])
            channel_analyses.append(ChannelAnalysis(
                channel=ch,
                dominant_freq=dom_freq,
                dominant_band=dom_band,
                alpha_power=ch_powers['alpha']/ch_total*100,
                theta_power=ch_powers['theta']/ch_total*100,
                delta_power=ch_powers['delta']/ch_total*100
            ))
        
        abnormalities = classify_background(
            rel_powers['alpha'], rel_powers['theta'],
            rel_powers['delta'], pdr
        )
        
        if asymmetry and asymmetry > 30:
            abnormalities.append(f"Significant interhemispheric asymmetry ({asymmetry:.0f}%)")
        
        ilae_grade = get_ilae_grade(
            pdr, rel_powers['delta'],
            rel_powers['theta'], rel_powers['alpha']
        )
        
        if not abnormalities:
            bg_class = "Normal background activity"
            summary = f"EEG shows normal background activity with posterior dominant rhythm of {pdr:.1f} Hz. No epileptiform discharges detected in this automated analysis." if pdr else "EEG shows normal background activity. No epileptiform discharges detected in this automated analysis."
        else:
            bg_class = "Abnormal background activity"
            summary = f"EEG shows abnormal background activity. Findings: {'; '.join(abnormalities)}. ILAE background grade: {ilae_grade}/5. Note: This is an automated analysis — clinical correlation and expert review required."
        
        return EEGAnalysis(
            duration_seconds=duration,
            num_channels=len(ch_names),
            sampling_rate=sfreq,
            channel_names=ch_names,
            global_band_powers=BandPower(**{k: rel_powers[k]*100 for k in bands}),
            posterior_dominant_rhythm=pdr,
            posterior_dominant_amplitude=pdr_amp,
            interhemispheric_asymmetry=asymmetry,
            background_classification=bg_class,
            abnormalities=abnormalities,
            channel_analyses=channel_analyses,
            ilae_background_grade=ilae_grade,
            summary=summary
        )
    
    finally:
        os.unlink(tmp_path)
