"""G-Cut seed proposal and image-reviewed identity; no neuron pruning."""
import hashlib,json
import numpy as np
from scipy import ndimage as ndi
from algorithms.tracing.gcut_adapter import component_maxima
VERSION='gsdt-gemma-soma-seeds-v1'

def fingerprint(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()

def gsdt_volume(raw,mask,percentile=55.):
    if raw.ndim!=3 or raw.shape!=mask.shape:raise ValueError('Expected matching resampled ZYX volumes')
    values=raw[mask>0]
    if not values.size:raise ValueError('Empty mask')
    threshold=float(np.percentile(values,percentile));binary=raw>threshold
    return threshold,binary,ndi.distance_transform_edt(binary)

def peaks(distance,threshold):
    labels,n=ndi.label(distance>threshold)
    return [dict(zyx=list(p),score=float(distance[p])) for p in component_maxima(distance,labels,n)]

def nms_reference(coords,scores,radius=6.):
    kept=[];xyz=np.asarray(coords)
    for i in np.argsort(scores)[::-1]:
        if all(np.linalg.norm(xyz[i]-xyz[j])>=radius for j in kept):kept.append(int(i))
    return [list(coords[i]) for i in kept]

def candidates(raw,mask,target_zyx,baseline=5.,relaxed=4.,percentile=55.):
    if not 0<relaxed<baseline:raise ValueError('Relaxed threshold must be positive and below baseline')
    target=np.array(target_zyx,float)
    if not np.all(np.isfinite(target)&(target>=0)&(target<raw.shape)):raise ValueError('Target outside volume')
    intensity,binary,distance=gsdt_volume(raw,mask,percentile)
    old=peaks(distance,baseline);new=peaks(distance,relaxed);points={}
    for threshold,items in [(baseline,old),(relaxed,new)]:
        for p in items:
            key=tuple(p['zyx']);v=points.setdefault(key,dict(zyx=p['zyx'],xyz_um=p['zyx'][::-1],gsdt_score=p['score'],threshold_sources=[]));v['threshold_sources'].append(threshold)
    # Exact-coordinate coalescing only. Old peaks survive if lowering the
    # component threshold joins two regions into one. No distance suppression.
    items=sorted(points.values(),key=lambda p:(-p['gsdt_score'],p['zyx']))
    for i,p in enumerate(items,1):p['candidate_id']=f'C{i:03d}'
    def baseline_points(ps):return nms_reference([target.tolist()]+[p['zyx'] for p in ps],[float('inf')]+[p['score'] for p in ps])
    return dict(policy=VERSION,shape_zyx=list(raw.shape),voxel_size_um_xyz=[1,1,1],target=dict(candidate_id='T',zyx=target.tolist(),xyz_um=target[::-1].tolist(),source='known_metadata_not_model_detection'),intensity_percentile=percentile,intensity_threshold=intensity,threshold_semantics='strict distance > threshold after raw > intensity percentile',baseline_threshold=baseline,relaxed_threshold=relaxed,baseline_gsdt=old,relaxed_gsdt=new,candidates=items,baseline_nms_zyx=baseline_points(old),relaxed_nms_zyx=baseline_points(new),nms_radius_um=6.,automatic_filters_applied=['exact_coordinate_coalescing_only'])

def validate(j,candidate,comparables,views):
    if not isinstance(j,dict):return ['not_object']
    errors=[]
    if j.get('candidate_id')!=candidate['candidate_id']:errors.append('invalid_candidate_id')
    soma=j.get('is_soma');identity=j.get('identity');action=j.get('decision');duplicate=j.get('duplicate_of')
    if soma not in ['yes','no','uncertain']:errors.append('invalid_is_soma')
    if identity not in ['distinct','same','unknown','not_applicable']:errors.append('invalid_identity')
    if action not in ['ACCEPT','DUPLICATE','REJECT','REVIEW']:errors.append('invalid_decision')
    if not isinstance(j.get('reason'),str) or not j['reason'].strip():errors.append('missing_reason')
    if action=='ACCEPT' and (soma!='yes' or identity!='distinct' or duplicate is not None):errors.append('inconsistent_accept')
    if action=='REJECT' and (soma!='no' or identity!='not_applicable' or duplicate is not None):errors.append('inconsistent_reject')
    if action=='DUPLICATE' and (soma!='yes' or identity!='same' or not isinstance(duplicate,str) or duplicate not in comparables):errors.append('invalid_duplicate')
    if action=='REVIEW' and duplicate is not None:errors.append('review_cannot_merge')
    evidence=j.get('evidence');planes=set();pair_planes=set()
    if not isinstance(evidence,list):return errors+['missing_evidence']
    for e in evidence:
        if not isinstance(e,dict) or not isinstance(e.get('view_id'),str) or e['view_id'] not in views:errors.append('invalid_view');continue
        v=views[e['view_id']]
        if not v['raw']:errors.append('not_raw_evidence')
        if not isinstance(e.get('observation'),str) or not e['observation'].strip():errors.append('empty_observation')
        if candidate['candidate_id'] in v['landmarks'] and v['landmarks'][candidate['candidate_id']].get('in_depth',True):planes.add(v['plane'])
        if isinstance(duplicate,str) and duplicate in v['landmarks'] and candidate['candidate_id'] in v['landmarks'] and v['landmarks'][duplicate].get('in_depth',True) and v['landmarks'][candidate['candidate_id']].get('in_depth',True):pair_planes.add(v['plane'])
    if action in ['ACCEPT','REJECT','DUPLICATE'] and not ('XY' in planes and planes&{'XZ','YZ'}):errors.append('missing_local_multiview_evidence')
    if action=='DUPLICATE' and not ('XY' in pair_planes and pair_planes&{'XZ','YZ'}):errors.append('missing_same_body_pair_evidence')
    return sorted(set(errors))

def review_fallback(candidate,reason):
    return dict(candidate_id=candidate['candidate_id'],is_soma='uncertain',identity='unknown',decision='REVIEW',duplicate_of=None,evidence=[],reason=reason)

def assemble(bundle,results):
    accepted=[dict(bundle['target'],model_verified=False)];mapping={};uncertain=[];rejected=[]
    known={'T'}
    for c in bundle['candidates']:
        if c['candidate_id'] not in results:raise ValueError('Incomplete review')
        r=results[c['candidate_id']];j=r['decision'];a=j['decision']
        if r['errors'] and a!='REVIEW':raise ValueError('Invalid decision cannot change seeds')
        if a=='ACCEPT':accepted.append(dict(c,model_verified=True));known.add(c['candidate_id'])
        elif a=='DUPLICATE':
            if j['duplicate_of'] not in known:raise ValueError('Duplicate must refer to an earlier retained soma')
            mapping[c['candidate_id']]=j['duplicate_of']
        elif a=='REJECT':rejected.append(c['candidate_id'])
        elif a=='REVIEW':uncertain.append(c['candidate_id'])
        else:raise ValueError('Unknown decision')
    return dict(accepted=accepted,duplicate_mapping=mapping,rejected=rejected,review=uncertain,downstream_executed=False)
