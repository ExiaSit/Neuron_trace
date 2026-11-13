"""Canonicalize an irrelevant identity field without changing image decisions."""
from copy import deepcopy
VERSION='non-soma-identity-canonicalization-v1'
def normalize(judgment):
    value=deepcopy(judgment);changes=[]
    if isinstance(value,dict) and value.get('decision')=='REJECT' and value.get('is_soma')=='no' and value.get('duplicate_of') is None and value.get('identity') in ['distinct','unknown']:
        changes.append(dict(field='identity',before=value['identity'],after='not_applicable',reason='An explicit non-soma rejection has no soma identity; decision, is_soma and image evidence are unchanged.'))
        value['identity']='not_applicable'
    return value,changes
