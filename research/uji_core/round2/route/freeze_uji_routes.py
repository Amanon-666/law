"""Metadata-only freeze of seven named UJI walk logfiles from IPIN 2016.

python freeze_uji_routes.py data_raw/ipin2016/archive.zip OUTPUT_DIRECTORY
No scores or model predictions. No release evaluation/ground-truth files are
opened. Query WIFI lines supply only their application timestamp prefix:
query AP identifiers and RSS values are never decoded or used. Historical
WIFI lines supply timestamp and BSSID (for input-catalog integrity) only.
"""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

ARCHIVE_SHA='9b78ab0aee9544450dc6351a0dad56dfa0b8e0b6f290fced05bbd8318f4a2791'
BASE='2016_IPIN_Competition_Track03/01-Logifles/01-Training/'
SITES={
 'UJITI':dict(history=['logfile_UJITI_R1_NEXUS5.txt'],query='logfile_UJITI_R2_NEXUS5.txt',role='development',phone='NEXUS5'),
 'UJIUB':dict(history=['logfile_UJIUB_R1n_S3.txt','logfile_UJIUB_R1r_S3.txt','logfile_UJIUB_R2n_S3.txt','logfile_UJIUB_R2r_S3.txt'],query='logfile_UJIUB_R3_S3.txt',role='reserved_future_route',phone='S3'),
}
BUDGETS=(1,3,5,10,20)
N_DRAWS=40
METRES_PER_UJI_UNIT=.766
EXPECTED={'UJITI':{1:(61,44),2:(114,47),3:(48,22)},'UJIUB':{0:(31,20),1:(133,49),2:(48,12),5:(130,21)}}


def sha(data):return hashlib.sha256(data).hexdigest()


def canon(obj):return (json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode('ascii')


def save(path,obj):path.write_bytes(canon(obj));return sha(path.read_bytes())


def filehash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return h.hexdigest()


def axis_hash(names):return sha(('\n'.join(sorted(names))+'\n').encode('ascii'))


def utc(x):return datetime.fromtimestamp(x,timezone.utc).isoformat()


def read_metadata(z,name,query):
    archive_member=BASE+name;checksum=hashlib.sha256();P=[];T={};start=None
    with z.open(archive_member) as f:
        for line_no,raw in enumerate(f,1):
            checksum.update(raw)  # opaque checksum covers the full original member
            if line_no==2:
                header=raw.decode('latin-1').split(': ',1)[1].strip().replace('GMT','').replace('CEST','+02:00').replace('CET','+01:00')
                start=datetime.strptime(header,'%a %b %d %H:%M:%S %z %Y').timestamp()
            if raw.startswith(b'POSI;'):
                v=raw.rstrip().rstrip(b';').split(b';');P.append((float(v[1]),float(v[3]),float(v[4]),int(float(v[5])),int(float(v[6])),line_no))
            elif raw.startswith(b'WIFI;'):
                sep=raw.index(b';',5);t=float(raw[5:sep]);row=T.setdefault(t,[line_no,0,set() if not query else None]);row[1]+=1
                if not query:
                    # Follow public_corpus.adapter.parse_log's BSSID field.
                    # The final dBm field and intermediate SSID are never decoded.
                    bssid=raw.split(b';',5)[4].decode('ascii').lower();row[2].add(bssid)
    if start is None:raise ValueError(name)
    accepted=[];apsets={};P.sort()
    for a,b in zip(P[:-1],P[1:]):
        if a[3:5]!=b[3:5] or not 0.<b[0]-a[0]<=120.:continue
        for t in sorted(k for k in T if a[0]<=k<b[0]):
            w=(t-a[0])/(b[0]-a[0]);lat=a[1]+w*(b[1]-a[1]);lon=a[2]+w*(b[2]-a[2]);first,count,aps=T[t]
            rowid=f'{Path(name).stem}@W{first}'
            record=dict(id=rowid,file=name,first_WIFI_line=first,app_s=t,capture_unix_s=start+t,lat=lat,lon=lon,
                        floor=a[3],building=a[4],POSI_lines=[a[5],b[5]],POSI_app_s=[a[0],b[0]],interpolation=w)
            accepted.append(record)
            if aps is not None:apsets[rowid]=aps
    if len({r['id'] for r in accepted})!=len(accepted):raise ValueError('Duplicated interpolated scan')
    return accepted,apsets,dict(member=archive_member,sha256=checksum.hexdigest(),zip_crc32=f'{z.getinfo(archive_member).CRC:08x}',
        creation_utc=utc(start),accepted_WIFI_scans=len(accepted),distinct_WIFI_app_times=len(T),POSI_landmarks=len(P),
        first_accepted=utc(min(r['capture_unix_s'] for r in accepted)),last_accepted=utc(max(r['capture_unix_s'] for r in accepted)))


def positions(rows):return {(r['lat'],r['lon']) for r in rows}


def draw_s(ss,site,floor,draw):
    groups=defaultdict(list)
    for r in ss:groups[r['lat'],r['lon']].append(r['id'])
    key=f'uji-route-v1/{site}/F{floor}/draw{draw}/';encode=lambda p:p[0].hex()+';'+p[1].hex()
    keys=sorted(groups,key=lambda p:(sha((key+'position/'+encode(p)).encode()),encode(p)))[:max(BUDGETS)]
    return [min(groups[p],key=lambda i:(sha((key+'scan/'+i).encode()),i)) for p in keys]


def main(archive,out):
    out.mkdir(parents=True,exist_ok=True)
    if filehash(archive)!=ARCHIVE_SHA:raise ValueError('Archive checksum mismatch')
    scans=[];events=[];member_info=[];sites={};counts=[]
    with ZipFile(archive) as z:
        for site,spec in SITES.items():
            hpool=[];known={}
            for file in spec['history']:
                data,ap,info=read_metadata(z,file,False);hpool.extend(data);known.update(ap);scans.extend(data);member_info.append(info)
            qpool,no_ap,info=read_metadata(z,spec['query'],True);assert not no_ap
            scans.extend(qpool);member_info.append(info)
            assert max(r['capture_unix_s'] for r in hpool)<min(r['capture_unix_s'] for r in qpool)
            observed_floors=sorted({r['floor'] for r in qpool});siteq=0
            for floor in observed_floors:
                q=[r for r in qpool if r['floor']==floor];s=[r for r in hpool if r['floor']==floor];h=[r for r in hpool if r['floor']!=floor]
                elig=bool(h) and len(positions(s))>=max(BUDGETS) and bool(q)
                if not elig:raise ValueError(f'Unexpected ineligible floor {site}/{floor}; verify prior geometry')
                if positions(q)&positions(s):raise AssertionError('Exact S/Q pool overlap')
                if len(q)!=len(positions(q)) or (len(positions(s)),len(q))!=EXPECTED[site][floor]:raise AssertionError('Geometry counts changed')
                if {r['building'] for r in h+s+q}.__len__()!=1:raise AssertionError('Mixed internal building IDs')
                siteq+=len(q);h=sorted(h,key=lambda r:(r['floor'],r['capture_unix_s'],r['id']));q=sorted(q,key=lambda r:(r['capture_unix_s'],r['id']));origin=min(h,key=lambda r:(r['capture_unix_s'],r['id']))
                lat0,lon0=origin['lat'],origin['lon'];east=111320.*math.cos(math.radians(lat0))
                def xy(r):return [(r['lon']-lon0)*east,(r['lat']-lat0)*111320.]
                hrows=defaultdict(list);h_ap=set()
                for r in h:hrows[int(r['floor'])].append(r['id']);h_ap.update(known[r['id']])
                srid={r['id']:r for r in s};sqdist=lambda p,y:(p[0]-y[0])**2+(p[1]-y[1])**2
                qxy=[xy(r) for r in q];sxy=[xy(r) for r in s];near=[math.sqrt(min(sqdist(a,v) for v in sxy)) for a in qxy]
                # Freeze registered geometry separately from source-row membership.
                geom=[(r['id'],xy(r)[0].hex(),xy(r)[1].hex()) for r in sorted(h+s+q,key=lambda x:x['id'])]
                draws=[]
                for draw in range(N_DRAWS):
                    sr=draw_s(s,site,floor,draw);assert len(set(sr))==20 and len(positions([srid[x] for x in sr]))==20
                    aps=set(h_ap);axes={}
                    for i,rid in enumerate(sr,1):
                        aps.update(known[rid]);
                        if i in BUDGETS:axes[str(i)]={'N':len(aps),'sha256':axis_hash(aps)}
                    draws.append(dict(draw=draw,ordered_S_scan_ids=sr,permitted_AP_axis=axes))
                assert len({tuple(d['ordered_S_scan_ids']) for d in draws})==40
                events.append(dict(site=site,role=spec['role'],floor=floor,device=spec['phone'],
                    historical_files=spec['history'],next_query_file=spec['query'],origin_from_H=origin['id'],lat0=lat0,lon0=lon0,
                    east_m_per_degree=east,north_m_per_degree=111320.,coordinate_frame_sha256=sha(canon(geom)),
                    H_by_floor=dict(hrows),H_only_AP_axis={'N':len(h_ap),'sha256':axis_hash(h_ap)},
                    eligible_target_S_ids=sorted(srid),ordered_Q_ids=[r['id'] for r in q],draws=draws))
                counts.append(dict(site=site,role=spec['role'],floor=floor,H_scans=len(h),H_floors=sorted(hrows),
                    eligible_S_scans=len(s),distinct_S_positions=len(positions(s)),Q_scans=len(q),distinct_Q_positions=len(positions(q)),
                    exact_S_pool_Q_overlap=0,min_dist_to_S_pool_metres=min(near),Q_within_half_metre_of_S_pool=sum(n<=.5 for n in near)))
            sites[site]=dict(**spec,total_Q=siteq)
    assert len(scans)==len({r['id'] for r in scans}) and len(events)==7 and sum(len(e['ordered_Q_ids']) for e in events)==215
    # gzip without filename or wall-clock timestamp to keep hashes reproducible.
    buf=canon(sorted(scans,key=lambda x:x['id']));dest=out/'scan_metadata.json.gz'
    with dest.open('wb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as stream:stream.write(buf)
    metadata_sha=filehash(dest);event_sha=save(out/'route_manifest.json',events)
    protocol=dict(design='UJI route-specific progression v1, metadata only; no RSS parse, predictions, outcome inspection or selection.',
      dataset_doi='10.5281/zenodo.2791530',input_archive_sha256=ARCHIVE_SHA,allowed_members=member_info,
      generator_sha256=filehash(Path(__file__)),scan_metadata_sha256=metadata_sha,scan_metadata_plain_sha256=sha(buf),manifest_sha256=event_sha,sites=sites,counts=counts,
      source_selection='All seven 2016 UJITI/UJIUB training logs in this release, no IPIN validation or competition evaluation files. Per site reserve the last whole route by creation date. UJITI R2 (Apr25 after Apr11 R1) is development. UJIUB R3 (Apr5 after March17/22 R1/R2 normal and return passes) is reserved for one prospective-route test following the scientific freeze. The design generator never decodes UJIUB query APs or RSS and does not compute outcomes.',
      H_S_Q='Known target floor. H = all temporally earlier valid scans at every other floor; one H source per physical floor. S = K target-floor locations in the same earlier route pool, R=1 original scan/location, nested budgets 1/3/5/10/20. Same immutable H/S/Q to all arms. Q = all valid WiFi acquisition timestamps from the future designated route on this floor; no query-coverage exclusion.',
      interpolation='Each WiFi scan is all WIFI records with the same exact application timestamp in one logfile. POSI sorted by application timestamp; accept a WiFi scan iff between adjacent same-floor, same-building endpoints, t_start<=t<t_end, 0<gap<=120 seconds. Linearly interpolate geodetic latitude/longitude. Preserve raw member, first WIFI line, POSI line pair, elapsed time/fraction for each scan. No waypoint counting as WiFi samples.',
      projection='For each target floor use earliest *allowed H* scan as a common origin. E=(lon-lon0)*111320*cos(lat0 radians), N=(lat-lat0)*111320 metres. Map/parameters use XY_m/0.766 then multiply predictions by 0.766 to score metre errors. Equality uses native interpolated binary float coordinate pairs. Frame hash = JSON of sorted scan-id/E.hex/N.hex in that target frame. No query fits coordinates, geometry, APs or frame.',
      AP_axis='Canonical sorted lowercase BSSID strings from H union only the current K S acquisitions. SHA256 of UTF-8/ASCII name list separated by LF and a final LF, without emitting BSSIDs. Future extraction uses final per-timestamp record for a duplicate BSSID, 100 for nondetection, and discards query-only BSSIDs. Q WIFI records here contribute timestamp and raw row-number only; Q AP and dBm fields not parsed. Row list and axis hash verified before any inference.',
      draws='40 deterministic SHA256 orderings of distinct S positions, lexical SHA strings over uji-route-v1/{site}/F{floor}/draw{draw}/position/{lat.hex()};{lon.hex()}. At a repeated position take smallest SHA from same prefix plus scan/{scan-id}. Max-20 retained and fixed prefixes by K. No seed tuning or query coordinates choose supports.',
      estimators='Existing unmodified gap-aware ReFi and registered learned/fixed controls must be pinned by SHA before any use in a future study. Direct strong WKNN on SAME inputs uses already frozen external [prototype,alpha=.35,k=3]; same-map control uses k=7 on identical calibrated prototypes, floor gaps abs(f_source-f_target). For different detected/missing template means, apply comparison with the same two means under the same H+S AP catalogue. No model scores in this freeze and no new model arm.',
      inference='Known floors. Primary mean Euclidean error in metres, average scan errors at each Q position, then positions, draws, floors within site (also report each floor and K). Always keep UJITI development separated from UJIUB reserved route. Nested K/draw comparisons pair on identical Q. Save all predictions and numerical failures. Many adjacent interpolated points on one path and 40 reused support sets are not independent visits. Same phone within each site: no held-device claim. Sites are known; separate trajectories and dates only. UJIUB is the university library; UJITI geometry previously compared to UJI B0. Verify that a 2017 re-release or prior study did not score the identical timestamped route before calling the route previously unseen. Do not claim a new campus/building or untouched original UJIIndoorLoc benchmark.',
      validation='215 distinct WiFi acquisitions and exact XYs across seven floor-specific future streams, zero target S-pool/Q exact-coordinate overlaps. 40 support orders, 5 K per floor. UJITI remains development. Score reserved UJIUB R3 only after prespecifying and committing the single candidate, comparators, evaluation metric, and acceptance criteria.')
    ph=save(out/'protocol.json',protocol);print(json.dumps(dict(protocol_sha256=ph,manifest_sha256=event_sha,scan_metadata_sha256=metadata_sha,generator_sha256=protocol['generator_sha256'],n_raw_scan_records=len(scans),Q_by_site={k:v['total_Q'] for k,v in sites.items()},floors=len(events),nested_tasks=len(events)*40*5,query_RSS_parsed=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();main(a.archive,a.output)
