import urllib.request
import urllib.parse
import json

for pid in ['151594008', '151932002', '152071010']:
    print("=" * 70)
    print(f"TESTING AUDITED PATIENT #{pid}")
    print("=" * 70)
    data = urllib.parse.urlencode({
        'preset_pid': pid,
        'demographics_mode': 'auto',
        'age': '55',
        'gender': '1'
    }).encode('utf-8')
    
    req = urllib.request.Request('http://127.0.0.1:5050/analyze', data=data, method='POST')
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        exp = res['explanation']
        print(f"CVD Risk:        {res['cvd_risk_prob']*100:.1f}% [{exp['risk_badge']}]")
        print(f"CIMT Thickness:  {res['predicted_cimt_mm']:.3f} mm [{exp['cimt_status']}]")
        print(f"Retinal Age:     {res['predicted_age']:.1f} yrs (Gap: {res['retinal_age_gap']:+.1f} yrs)")
        print(f"Biological Sex:  {res['predicted_gender']} ({res['gender_confidence']*100:.1f}%)")
        print(f"\n[Microvascular Morphological Finding]:\n{exp['bio_explanation']}")
        print(f"\n[Plain-Language (Layman) Summary]:\n{exp['layman_summary']}")
        print(f"\n[Recommended Action & Referral]:\n{exp['recommendation']}\n")
