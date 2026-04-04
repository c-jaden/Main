import zipfile, io, requests, pandas as pd

url = "https://nces.ed.gov/ccd/Data/zip/ccd_sch_029_2425_w_1a_073025.zip"
resp = requests.get(url, timeout=120)
zf = zipfile.ZipFile(io.BytesIO(resp.content))
for name in zf.namelist():
    if name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(zf.read(name)), encoding="latin-1",
                         nrows=1, dtype=str)
        print("\n".join(sorted(df.columns.tolist())))
        break